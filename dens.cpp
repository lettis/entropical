
#include <algorithm>
#include <omp.h>

#include "dens.hpp"
#include "tools.hpp"

namespace Dens {

  std::vector<float>
  compute_densities(Tools::OCL::GPUElement* gpu
                  , float* coords
                  , std::size_t n_rows
                  , std::size_t i_col
                  , float h
                  , std::size_t n_wg
                  , std::size_t wgsize) {
    using Tools::OCL::check_error;
    float h_inv = 1.0f / h;
    // pre-sort coordinates
    std::vector<float> sorted_coords(n_rows);
    for (std::size_t i=0; i < n_rows; ++i) {
      sorted_coords[i] = coords[i_col*n_rows + i];
    }
    std::sort(sorted_coords.begin(), sorted_coords.end());
    // copy coords of given column to device
    check_error(clEnqueueWriteBuffer(gpu->q
                                   , gpu->buffers["sorted_coords"]
                                   , CL_TRUE
                                   , 0
                                   , sizeof(float) * n_rows
                                   , sorted_coords.data()
                                   , 0
                                   , NULL
                                   , NULL)
              , "clEnqueueWriteBuffer");
    check_error(clFlush(gpu->q), "clFlush");

    std::size_t global_worksize = n_wg * wgsize;
    std::size_t local_worksize = wgsize;

    Tools::OCL::set_kernel_arg(gpu, "initialize_zero", 0, std::string("P"));
    Tools::OCL::set_kernel_arg(gpu, "initialize_zero", 1, (unsigned int) n_rows);
    check_error(clEnqueueNDRangeKernel(gpu->q
                                     , gpu->kernels["initialize_zero"]
                                     , 1
                                     , NULL
                                     , &global_worksize
                                     , &local_worksize
                                     , 0
                                     , NULL
                                     , NULL)
              , "clEnqueueNDRangeKernel");
    check_error(clFlush(gpu->q), "clFlush");

    // run kernel-loop over all frames
    for (unsigned int i=0; i < n_rows; ++i) {
//      unsigned int n_wg_tmp = n_wg;
//      Tools::OCL::set_kernel_arg<std::string>(gpu, "initialize_zero", 0, std::string("P_partial"));
//      Tools::OCL::set_kernel_arg(gpu, "initialize_zero", 1, (unsigned int) n_wg);
//      check_error(clSetKernelArg(gpu->kernels["initialize_zero"]
//                               , 0
//                               , sizeof(cl_mem)
//                               , (void*) &(gpu->buffers["P_partial"]))
//                , "clSetKernelArg");
//      check_error(clSetKernelArg(gpu->kernels["initialize_zero"]
//                               , 1
//                               , sizeof(unsigned int)
//                               , &n_wg_tmp)
//                , "clSetKernelArg");
      check_error(clEnqueueNDRangeKernel(gpu->q
                                       , gpu->kernels["initialize_zero"]
                                       , 1
                                       , NULL
                                       , &global_worksize
                                       , &local_worksize
                                       , 0
                                       , NULL
                                       , NULL)
                , "clEnqueueNDRangeKernel");
      check_error(clFlush(gpu->q), "clFlush");
//      check_error(clFinish(gpu->q), "clFinish");
//
//      float ref_scaled_neg = -1.0f * coords[i_col*n_rows + i] * h_inv;
//      Tools::OCL::set_kernel_arg(gpu, "probs_1d", 0, std::string("sorted_coords"));
//      Tools::OCL::set_kernel_arg(gpu, "probs_1d", 1, (unsigned int) n_rows);
//      Tools::OCL::set_kernel_arg(gpu, "probs_1d", 2, std::string("P_partial"));
//      Tools::OCL::set_kernel_arg(gpu, "probs_1d", 3, std::string("P"));
//      Tools::OCL::set_kernel_arg(gpu, "probs_1d", 4, h_inv);
//      Tools::OCL::set_kernel_arg(gpu, "probs_1d", 5, ref_scaled_neg);
//      Tools::OCL::set_kernel_arg(gpu, "probs_1d", 6, i);
//      check_error(clEnqueueNDRangeKernel(gpu->q
//                                       , gpu->kernels["probs_1d"]
//                                       , 1
//                                       , NULL
//                                       , &global_worksize
//                                       , &local_worksize
//                                       , 0
//                                       , NULL
//                                       , NULL)
//                , "clEnqueueNDRangeKernel");
//      check_error(clFlush(gpu->q), "clFlush");
//      check_error(clFinish(gpu->q), "clFinish");
    }
    // retrieve probability densities from device
    std::vector<float> densities(n_rows);
    check_error(clEnqueueReadBuffer(gpu->q
                                  , gpu->buffers["P"]
                                  , CL_TRUE
                                  , 0
                                  , sizeof(float) * n_rows
                                  , densities.data()
                                  , 0
                                  , NULL
                                  , NULL)
              , "clEnqueueReadBuffer");
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
    std::vector<float> bandwidths;
    using Tools::String::split;
    for (std::string h: split(args["bandwidths"].as<std::string>()
                            , ' '
                            , true)) {
      bandwidths.push_back(std::stof(h));
    }
    if (bandwidths.size() != selected_cols.size()) {
      std::cerr << "error: number of given bandwidth values does not match"
                         " the number of selected columns!" << std::endl;
      exit(EXIT_FAILURE);
    }
    // setup OpenCL environment
    std::vector<Tools::OCL::GPUElement> gpus = Tools::OCL::gpus();
    std::size_t n_gpus = gpus.size();
    if (n_gpus == 0) {
      Tools::OCL::print_no_gpus_errmsg();
      exit(EXIT_FAILURE);
    }
    // determine workgroup size and no. of workgroups from
    // available device memory and data size
    unsigned int wgsize = Tools::OCL::max_wgsize(&gpus[0], sizeof(float));
    unsigned int n_wg = (unsigned int) std::ceil(n_rows / ((float) wgsize));
    //TODO embed kernel source in header
    std::string kernel_src = Tools::OCL::load_kernel_source("kernels.cl");
    for (unsigned int i=0; i < n_gpus; ++i) {
      Tools::OCL::setup_gpu(&gpus[i]
                          , kernel_src
                          , {"probs_1d", "initialize_zero"}
                          , wgsize);
      Tools::OCL::create_buffer(&gpus[i]
                              , "sorted_coords"
                              , sizeof(float) * n_wg * wgsize
                              , CL_MEM_READ_ONLY);
      Tools::OCL::create_buffer(&gpus[i]
                              , "P"
                              , sizeof(float) * n_rows
                              , CL_MEM_WRITE_ONLY);
      Tools::OCL::create_buffer(&gpus[i]
                              , "P_partial"
                              , sizeof(float) * n_wg
                              , CL_MEM_READ_WRITE);
    }
    // compute local densities on GPUs
    unsigned int i, j, thread_id;
    unsigned int n_selected_cols = selected_cols.size();
    std::vector<std::vector<float>> densities(selected_cols.size());
    #pragma omp parallel for default(none)\
                             private(i,j,thread_id)\
                             firstprivate(n_selected_cols,n_rows,\
                                          n_cols,n_wg,wgsize)\
                             shared(coords,selected_cols,gpus,\
                                    densities,bandwidths)\
                             num_threads(n_gpus)\
                             schedule(dynamic,1)
    for (j=0; j < n_selected_cols; ++j) {
      thread_id = omp_get_thread_num();
      densities[j] = compute_densities(&gpus[thread_id]
                                     , coords
                                     , n_rows
                                     , j
                                     , bandwidths[j]
                                     , n_wg
                                     , wgsize);
    }
//TODO normalized densities!
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

