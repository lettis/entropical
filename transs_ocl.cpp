
#include "transs_ocl.hpp"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>


int main_ocl(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS] FILE\n\n"
    "compute transfer entropies between principal components"));
  cl::Program prog;
  std::vector<cl::Device> devices;
  try {
    opts.add_options()
      // required options
      ("input,i", po::value<std::string>()->required(), "principal components (required).")
      ("tau,t", po::value<unsigned int>()->default_value(1), "lagtime (in # frames; default: 1).")
      // optional parameters
      ("pcmax", po::value<unsigned int>()->default_value(0), "max. PC to read (default: 0 == read all)")
      ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
      ("wgsize", po::value<unsigned int>()->default_value(128), "OpenCL workgroup size (must be multiple of 32; default: 128)")
      ("ngpu", po::value<unsigned int>()->default_value(1), "number of GPUs to use (default: 1) ATTENTION: MULTI-GPU NOT YET IMPLEMENTED.")
      ("help,h", po::bool_switch()->default_value(false), "show this help.");
    // option parsing, settings, checks
    po::positional_options_description pos_opts;
    pos_opts.add("input", 1);
    po::store(po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run(), args);
    po::notify(args);
    if (args["help"].as<bool>()) {
      std::cout << opts << std::endl;
      return EXIT_SUCCESS;
    }
    // set output stream to file or STDOUT, depending on args
    Tools::IO::set_out(args["output"].as<std::string>()); 
    std::string fname_input = args["input"].as<std::string>();
    unsigned int pc_max = args["pcmax"].as<unsigned int>();
    if (pc_max == 1) {
      std::cerr << "error: need at least two PCs to compute information transfer" << std::endl;
      return EXIT_FAILURE;
    }
    unsigned int tau = args["tau"].as<unsigned int>();
    if (tau == 0) {
      std::cerr << "error: a lagtime of 0 frames does not make sense." << std::endl;
      return EXIT_FAILURE;
    }
    unsigned int wgsize = args["wgsize"].as<unsigned int>();
    if (wgsize == 0) {
      std::cerr << "error: with a workgroup size of 0 work items, nothing can be computed." << std::endl;
      return EXIT_FAILURE;
    } else if (wgsize % 32 != 0) {
      std::cerr << "error: workgroup size must be a multiple of 32." << std::endl;
      return EXIT_FAILURE;
    }
    unsigned int ngpu = args["ngpu"].as<unsigned int>();
    if (wgsize == 0) {
      std::cerr << "error: with no GPUs to use, how should I compute anything?" << std::endl;
      return EXIT_FAILURE;
    }
    // read coordinates
    TRANSS_FLOAT* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    if (pc_max == 0) {
      // read full data set
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<TRANSS_FLOAT>(fname_input, 'C');
    } else {
      // read only until given PC
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<TRANSS_FLOAT>(fname_input, 'C', Tools::range<std::size_t>(0, pc_max, 1));
    }
    // compute sigmas and (first and second) minimal values for every dimension
    std::vector<TRANSS_FLOAT> sigmas(n_cols);
    std::vector<TRANSS_FLOAT> p_s(n_cols);
    std::vector<TRANSS_FLOAT> col_min1(n_cols, std::numeric_limits<TRANSS_FLOAT>::infinity());
    std::vector<TRANSS_FLOAT> col_min2(n_cols, std::numeric_limits<TRANSS_FLOAT>::infinity());
    {
      using namespace boost::accumulators;
      using VarAcc = accumulator_set<TRANSS_FLOAT, features<tag::variance(lazy)>>;
      for (std::size_t j=0; j < n_cols; ++j) {
        VarAcc acc;
        for (std::size_t i=0; i < n_rows; ++i) {
          TRANSS_FLOAT current_value = coords[j*n_rows+i];
          acc(current_value);
          col_min1[j] = std::min(col_min1[j], current_value);
          if (col_min1[j] < current_value && current_value < col_min2[j]) {
            col_min2[j] = current_value;
          }
        }
        // collect resulting sigmas from accumulators
        sigmas[j] = sqrt(variance(acc));
        p_s[j] = -1.0f / (2*pow(n_rows, -2.0/7.0)*sigmas[j]*sigmas[j]);
      }
    }
    // initialize OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM
                                  , (cl_context_properties)(platforms[0])()
                                  , 0 };
    cl::Context ctx(CL_DEVICE_TYPE_GPU, cps);
    devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue q(ctx, devices[0]);
    // kernel partial_sums
    //   build options to set:
    //     TAU:    the tau value (some unsigned integer > 0)
    //     N_ROWS: number of rows in input (== to size of x and y)
    //     WGSIZE: local size, i.e. number of worker items in workgroup
    //     PCMAX:  highest PC to take into account
    std::string src_partial_sums = 

                      //"#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
                      "#define TRANSS_FLOAT float\n"
                      "#define TRANSS_FLOAT4 float4\n"
                      "#define TRANSS_FLOAT_K (-16.11809565f)\n"

                      "#define POW2(X) (X)*(X)\n"
                      "#define TWO_PI 6.283185307179586\n"

                      "  __kernel void\n"
                      "  partial_sums(uint n\n"
                      "             , TRANSS_FLOAT p_s_y\n"
                      "             , TRANSS_FLOAT p_s_x\n"
                      "             , __global const TRANSS_FLOAT* y\n"
                      "             , __global const TRANSS_FLOAT* x\n"
                      "             , uint iy\n"
                      "             , uint ix\n"
                      "             , __global TRANSS_FLOAT4* S\n"
                      "             , TRANSS_FLOAT m_y\n"
                      "             , TRANSS_FLOAT m_x\n"
                      "             , __global TRANSS_FLOAT* T) {\n"
                      "    __local TRANSS_FLOAT4 S_loc[WGSIZE];\n"
                      "    uint gid = get_global_id(0);\n"
                      "    uint lid = get_local_id(0);\n"
                      "    uint wid = get_group_id(0);\n"
                      "    uint group_size = get_local_size(0);\n"
                      "    uint n_workgroups = get_num_groups(0);\n"
                                                                    // compute local values
                      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                      "    if (gid < N_ROWS) {\n"
                      "      TRANSS_FLOAT x_n = x[n];\n"
                      "      TRANSS_FLOAT x_ntau = x[n+TAU];\n"
                      "      TRANSS_FLOAT x_i = x[gid];\n"
                      "      TRANSS_FLOAT y_n = y[n];\n"
                      "      TRANSS_FLOAT y_i = y[gid];\n"
                      "      TRANSS_FLOAT s_y = p_s_y * POW2(y_n-y_i);\n"
                      "      TRANSS_FLOAT s_x = p_s_x * POW2(x_n-x_i);\n"
                      "      TRANSS_FLOAT s_x_ntau = p_s_x * POW2(x_ntau-x_i);\n"
                      "      s_y = (s_y - m_y < TRANSS_FLOAT_K) ? 0.0f : exp(s_y);\n"
                      "      s_x = (s_x - m_x < TRANSS_FLOAT_K) ? 0.0f : exp(s_x);\n"
                      "      s_x_ntau = (s_x_ntau - m_x < TRANSS_FLOAT_K) ? 0.0f : exp(s_x_ntau);\n"
                      "      S_loc[lid] = (TRANSS_FLOAT4) (s_x\n"
                      "                                  , s_x_ntau * s_x * s_y\n"
                      "                                  , s_x * s_y\n"
                      "                                  , POW2(s_x));\n"
                      "    } else {\n"
                      "      S_loc[lid] = (TRANSS_FLOAT4) (0.0f, 0.0f, 0.0f, 0.0f);\n"
                      "    }\n"
                                                                    // accumulate S locally
                      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
//                      "    if (lid == 0) {\n"
//                      "      TRANSS_FLOAT4 S_acc = S_loc[0];\n"
//                      "      for (uint i=1; i < WGSIZE; ++i) {\n"
//                      "        S_acc += S_loc[i];\n"
//                      "      }\n"
//                      "      S[wid] = S_acc;\n"
//                      "    }\n"
                      "    for(int i=group_size/2; i > 0; i>>=1) { \n"
                      "      if (lid < i) {\n"
                      "        S_loc[lid] += S_loc[lid+i];
                      "      }\n"
                      "      barrier(CLK_LOCAL_MEM_FENCE);\n"
                      "    }\n"
                      "    if (lid == 0) {\n"
                      "      S[wid] = S_loc[0];\n"
                      "    }\n"
                                                                    // accumulate S globally
//                      "    if (gid == 0) {\n"
//                      "      TRANSS_FLOAT4 S_acc = (TRANSS_FLOAT4) (0.0f, 0.0f, 0.0f, 0.0f);\n"
//                      "      for (uint i=0; i < n_workgroups; ++i) {\n"
//                      "        S_acc += S[i];\n"
//                      "      }\n"
//                      "      S_acc.s0 = log(TWO_PI * S_acc.s1 * S_acc.s0 / S_acc.s2 / S_acc.s3);\n"
//                      "      S_acc.s1 = exp(log(S_acc.s1) + 2*m_x + m_y);\n"
//                      "      T[iy*PCMAX+ix] += S_acc.s0 * S_acc.s1;\n"
//                      "    }\n"
                      "    barrier(CLK_GLOBAL_MEM_FENCE);\n"
                      "    for(int i=n_workgroups/2; i > 0; i>>=1) { \n"
                      "      if (lid == 0 && wid < i) {\n"
                      "        S[wid] += S[wid+i];
                      "      }\n"
                      "      barrier(CLK_GLOBAL_MEM_FENCE);\n"
                      "    }\n"
                      "    if (lid == 0 && wid == 0) {\n"
                      "      S[0].s0 = log(TWO_PI * S[0].s1 * S[0].s0 / S[0].s2 / S[0].s3);\n"
                      "      S[0].s1 = exp(log(S[0].s1) + 2*m_x + m_y);\n"
                      "      T[iy*PCMAX+ix] += S[0].s0 * S[0].s1;\n"
                      "    }\n"
                      "  }\n"
    ;
    cl::Program::Sources src(1, std::make_pair(src_partial_sums.c_str(), src_partial_sums.length()+1));
    prog = cl::Program(ctx, src);
    prog.build(devices
             , Tools::String::printf("-D TAU=%d -D N_ROWS=%d -D WGSIZE=%d -D PCMAX=%d", tau, n_rows, wgsize, pc_max).c_str());
    cl::Kernel knl(prog, "partial_sums");
    // workload grid dimensions
    unsigned int n_workgroups = n_rows / wgsize;
    if (n_rows % wgsize != 0) {
      ++n_workgroups;
    }
    unsigned int n_total_workitems = n_workgroups * wgsize;
    cl::NDRange global(n_total_workitems);
    cl::NDRange local(wgsize);
    // set up host/device buffers
    std::vector<TRANSS_FLOAT> T(pc_max*pc_max, 0.0f);
    cl::Buffer y_buf(ctx, CL_MEM_READ_ONLY, n_rows*sizeof(TRANSS_FLOAT));
    cl::Buffer x_buf(ctx, CL_MEM_READ_ONLY, n_rows*sizeof(TRANSS_FLOAT));
    cl::Buffer S_buf(ctx, CL_MEM_READ_WRITE, n_workgroups*sizeof(TRANSS_FLOAT)*4);
    cl::Buffer T_buf(ctx, CL_MEM_READ_WRITE, pc_max*pc_max*sizeof(TRANSS_FLOAT));
    q.enqueueWriteBuffer(T_buf, CL_TRUE, 0, pc_max*pc_max*sizeof(TRANSS_FLOAT), T.data());
    // set default arguments for kernel
    knl.setArg(7, S_buf);
    knl.setArg(10, T_buf);
    // run computation
    for (unsigned int iy=0; iy < pc_max; ++iy) {
      q.enqueueWriteBuffer(y_buf, CL_TRUE, 0, n_rows*sizeof(TRANSS_FLOAT), &coords[iy*n_rows]);
      TRANSS_FLOAT m_y = p_s[iy] * POW2(col_min2[iy] - col_min1[iy]);
      // off-diagonals
      for (unsigned int ix=iy+1; ix < pc_max; ++ix) {
        q.enqueueWriteBuffer(x_buf, CL_TRUE, 0, n_rows*sizeof(TRANSS_FLOAT), &coords[ix*n_rows]);
        TRANSS_FLOAT m_x = p_s[ix] * POW2(col_min2[ix] - col_min1[ix]);
        // y -> x
        knl.setArg(1, p_s[iy]);
        knl.setArg(2, p_s[ix]);
        knl.setArg(3, y_buf);
        knl.setArg(4, x_buf);
        knl.setArg(5, iy);
        knl.setArg(6, ix);
        knl.setArg(8, m_y);
        knl.setArg(9, m_x);
        for (unsigned int n=0; n < n_rows-tau; ++n) {
          knl.setArg(0, n);
          q.enqueueNDRangeKernel(knl, cl::NullRange, global, local);
        }
        // x -> y
        knl.setArg(1, p_s[ix]);
        knl.setArg(2, p_s[iy]);
        knl.setArg(3, x_buf);
        knl.setArg(4, y_buf);
        knl.setArg(5, ix);
        knl.setArg(6, iy);
        knl.setArg(8, m_x);
        knl.setArg(9, m_y);
        for (unsigned int n=0; n < n_rows-tau; ++n) {
          knl.setArg(0, n);
          q.enqueueNDRangeKernel(knl, cl::NullRange, global, local);
        }
      }
      // diagonals
      knl.setArg(1, p_s[iy]);
      knl.setArg(2, p_s[iy]);
      knl.setArg(3, y_buf);
      knl.setArg(4, y_buf);
      knl.setArg(5, iy);
      knl.setArg(6, iy);
      knl.setArg(8, m_y);
      knl.setArg(9, m_y);
      for (unsigned int n=0; n < n_rows-tau; ++n) {
        knl.setArg(0, n);
        q.enqueueNDRangeKernel(knl, cl::NullRange, global, local);
      }
    }
    // retrieve results
    q.enqueueReadBuffer(T_buf, CL_TRUE, 0, pc_max*pc_max*sizeof(TRANSS_FLOAT), T.data());
    // clean up
    Tools::IO::free_coords(coords);
    // output
    for (unsigned int iy=0; iy < n_cols; ++iy) {
      for (unsigned int ix=0; ix < n_cols; ++ix) {
        Tools::IO::out() << " "
//                         << T[iy*pc_max+ix] * pow(n_rows, -4.0/7.0) / (pow(2*M_PI, 3.0/2.0) * sigmas[iy]*sigmas[ix]*sigmas[ix]);
                         << T[iy*pc_max+ix];
      }
      Tools::IO::out() << "\n";
    }
  } catch (const boost::bad_any_cast& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments!\n\n";
    }
    std::cerr << opts << std::endl;
    return EXIT_FAILURE;
  } catch (cl::Error e) {
    std::cerr << e.what() << ": Error code " << e.err() << std::endl;
    std::cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "\n" << e.what() << "\n\n";
    std::cerr << opts << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

