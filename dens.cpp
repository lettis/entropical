
#include <omp.h>

#include "dens.hpp"
#include "tools.hpp"

namespace Dens {

  void
  setup_gpu(GPUElement& gpu
          , std::string kernel_src
          , unsigned int n_rows) {

    //TODO adapt to Dens


    unsigned int n_extended = n_workgroups * wgsize;
    cl_int err;
    // prepare kernel source
    kernel_src = std::string("#define WGSIZE ")
                 + std::to_string(wgsize)
                 + std::string("\n")
                 + kernel_src;
    const char* src = kernel_src.c_str();
    // create context
    gpu.ctx = clCreateContext(NULL
                            , 1
                            , &gpu.i_dev
                            , &pfn_notify
                            , NULL
                            , &err);
    check_error(err, "clCreateContext");
    // command queue
    gpu.q = clCreateCommandQueue(gpu.ctx
                               , gpu.i_dev
                               , 0
                               , &err);
    check_error(err, "clCreateCommandQueue");
    // create program
    gpu.prog = clCreateProgramWithSource(gpu.ctx
                                       , 1
                                       , &src
                                       , NULL
                                       , &err);
    check_error(err, "clCreateProgramWithSource");
    // compile kernels
    if (clBuildProgram(gpu.prog
                     , 1
                     , &gpu.i_dev
                     , ""
                     , NULL
                     , NULL) != CL_SUCCESS) {
      char buffer[10240];
      clGetProgramBuildInfo(gpu.prog
                          , gpu.i_dev
                          , CL_PROGRAM_BUILD_LOG
                          , sizeof(buffer)
                          , buffer
                          , NULL);
      std::cerr << "CL Compilation failed:" << std::endl
                << buffer << std::endl;
      exit(EXIT_FAILURE);
    }
    // create kernel objects
    auto create_kernel = [&](std::string kname) -> void {
      gpu.kernels[kname] = clCreateKernel(gpu.prog
                                        , kname.c_str()
                                        , &err);
      check_error(err, "clCreateKernel");
    };
    create_kernel("partial_probs");
    create_kernel("collect_partials");
    create_kernel("compute_T");
    create_kernel("initialize_zero");
    // create buffers
    auto create_buffer = [&](std::string bname
                           , std::size_t bsize
                           , cl_mem_flags bflags) -> void {
      gpu.buffers[bname] = clCreateBuffer(gpu.ctx
                                        , bflags
                                        , bsize
                                        , NULL
                                        , &err);
      check_error(err, "clCreateBuffer");
    };
    // input coordinates
    create_buffer("i"
                , sizeof(float) * n_extended
                , CL_MEM_READ_WRITE);
    // input coordinates
    create_buffer("j"
                , sizeof(float) * n_extended
                , CL_MEM_READ_WRITE);
    // probabilities pre-reduced on workgroup level (single frame)
    create_buffer("Psingle"
                , sizeof(float) * 4*n_workgroups
                , CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
    // partially accumulated probabilities
    create_buffer("Pacc_partial"
                , sizeof(float) * 4*n_rows
                , CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
    // partially accumulated transfer entropy
    create_buffer("Tacc_partial"
                , sizeof(float) * n_rows
                , CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
    // resulting transfer entropy
    create_buffer("T"
                , sizeof(float) * 2
                , CL_MEM_READ_WRITE);
  }




  std::vector<float>
  compute_densities(Tools::OCL::GPUElement& gpu
                  , float* coords
                  , std::size_t n_rows
                  , std::size_t n_cols
                  , std::size_t i_col) {
    std::vector<float> densities(n_rows);

    //TODO compute densities

    return densities;
  }

  void
  main(boost::program_options::variables_map args) {
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    // read data
    std::vector<std::size_t> selected_cols;
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::tie(selected_cols, coords, n_rows, n_cols)
      = Tools::IO::selected_coords<float>(fname_input
                                        , args["columns"].as<std::string>());
    // setup OpenCL environment
    std::vector<Tools::OCL::GPUElement> gpus = Tools::OCL::gpus();
    std::size_t n_gpus = gpus.size();
    if (n_gpus == 0) {
      Tools::OCL::print_no_gpus_errmsg();
      exit(EXIT_FAILURE);
    }
    //TODO finish OpenCL setup
    // compute local densities on GPUs
    unsigned int i, j, thread_id;
    unsigned int n_selected_cols = selected_cols.size();
    std::vector<std::vector<float>> densities(selected_cols.size());
    #pragma omp parallel for default(none)\
                             private(j,thread_id)\
                             firstprivate(n_selected_cols,n_rows,n_cols)\
                             shared(coords,selected_cols,gpus,densities)\
                             schedule(dynamic,1)
    for (j=0; j < n_selected_cols; ++j) {
      thread_id = omp_get_thread_num();
      densities[j] = compute_densities(gpus[thread_id]
                                     , coords
                                     , n_rows
                                     , n_cols
                                     , selected_cols[j]);
    }
    // output: densities
    for (i=0; i < n_rows; ++i) {
      for (j=0; j < selected_cols.size(); ++j) {
        Tools::IO::out() << " " << densities[j][i];
      }
      Tools::IO::out() << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Dens'

