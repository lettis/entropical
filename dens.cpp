
#include <algorithm>
#include <omp.h>

#include "dens.hpp"
#include "tools.hpp"

namespace Dens {

  std::vector<float>
  compute_densities_1d(Tools::OCL::GPUElement* gpu
                     , const float* coords
                     , std::size_t n_rows
                     , std::size_t i_col
                     , float h
                     , std::size_t n_wg
                     , std::size_t wgsize) {
    using Tools::OCL::check_error;
    float h_inv = 1.0f / h;
    // helper functions to run kernel
    auto nq_range_offset = [&] (std::string kname
                              , std::size_t offset
                              , std::size_t range) {
      check_error(clEnqueueNDRangeKernel(gpu->q
                                       , gpu->kernels[kname]
                                       , 1
                                       , &offset
                                       , &range
                                       , &wgsize
                                       , 0
                                       , NULL
                                       , NULL)
                , "clEnqueueNDRangeKernel");
    };
    // pre-sort coordinates
    std::vector<float> sorted_coords(n_rows);
    for (std::size_t i=0; i < n_rows; ++i) {
      sorted_coords[i] = coords[i_col*n_rows + i];
    }
    std::sort(sorted_coords.begin(), sorted_coords.end());
    // limits for box-assisted pruning
    std::vector<float> boxlimits = Tools::boxlimits(sorted_coords, wgsize);
    // copy (sorted) coords of given column to device
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
    // run kernel-loop over all frames
    Tools::OCL::set_kernel_buf_arg(gpu
                                 , "partial_probs_1d"
                                 , 0
                                 , "sorted_coords");
    Tools::OCL::set_kernel_scalar_arg(gpu
                                    , "partial_probs_1d"
                                    , 1
                                    , (unsigned int) n_rows);
    Tools::OCL::set_kernel_buf_arg(gpu
                                 , "partial_probs_1d"
                                 , 2
                                 , "P_partial");
    Tools::OCL::set_kernel_scalar_arg(gpu
                                    , "partial_probs_1d"
                                    , 3
                                    , h_inv);
    Tools::OCL::set_kernel_buf_arg(gpu
                                 , "sum_partial_probs_1d"
                                 , 0
                                 , "P_partial");
    Tools::OCL::set_kernel_buf_arg(gpu
                                 , "sum_partial_probs_1d"
                                 , 1
                                 , "P");
    for (unsigned int i=0; i < n_rows; ++i) {
      // prune full range to limit of bandwidth
      float ref_val = coords[i_col*n_rows + i];
      auto min_max = Tools::min_max_box(boxlimits, ref_val, h);
      std::size_t mm_range = min_max.second - min_max.first + 1;

      // compute partials in workgroups
      float ref_scaled_neg = -1.0f * h_inv * ref_val;
      Tools::OCL::set_kernel_scalar_arg(gpu
                                      , "partial_probs_1d"
                                      , 4
                                      , ref_scaled_neg);
      nq_range_offset("partial_probs_1d"
                    , min_max.first * wgsize
                    , mm_range * wgsize);

      // stagewise reduction of workgroup results
      Tools::OCL::set_kernel_scalar_arg(gpu
                                      , "sum_partial_probs_1d"
                                      , 2 // i_ref
                                      , i);
      Tools::OCL::set_kernel_scalar_arg(gpu
                                      , "sum_partial_probs_1d"
                                      , 3 // n_partials
                                      , (unsigned int) mm_range);
      Tools::OCL::set_kernel_scalar_arg(gpu
                                      , "sum_partial_probs_1d"
                                      , 4 // n_wg
                                      , (unsigned int) n_wg);
      unsigned int rng = Tools::min_multiplicator(mm_range, wgsize) * wgsize;
      while (rng >= wgsize) {
        nq_range_offset("sum_partial_probs_1d"
                      , 0
                      , rng);
        rng /= wgsize;
      }
      // run queued kernels
      check_error(clFlush(gpu->q), "clFlush");
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
    float sum = Tools::kahan_sum(densities);
    for (float& d: densities) {
      d /= sum;
    }
    return densities;
  }


  std::vector<std::vector<float>>
  compute_densities(std::vector<std::size_t> selected_cols
                  , const float* coords
                  , std::size_t n_rows
                  , std::vector<float> bandwidths) {
    std::vector<std::vector<float>> densities(selected_cols.size());
    // setup OpenCL environment
    std::vector<Tools::OCL::GPUElement> gpus = Tools::OCL::gpus();
    std::size_t n_gpus = gpus.size();
    if (n_gpus == 0) {
      Tools::OCL::print_no_gpus_errmsg();
      exit(EXIT_FAILURE);
    }
    // determine workgroup size and no. of workgroups from
    // available device memory and data size
    unsigned int wgsize = Tools::OCL::max_wgsize(&gpus[0], sizeof(float)) / 4;
    unsigned int n_wg = Tools::min_multiplicator(n_rows, wgsize);
    unsigned int partial_size = Tools::min_multiplicator(n_wg, wgsize) * wgsize;
    //TODO embed kernel source in header
    std::string kernel_src = Tools::OCL::load_kernel_source("kernels.cl");
    for (unsigned int i=0; i < n_gpus; ++i) {
      Tools::OCL::setup_gpu(&gpus[i]
                          , kernel_src
                          , {"partial_probs_1d"
                           , "initialize_zero"
                           , "sum_partial_probs_1d"}
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
                              , sizeof(float) * partial_size
                              , CL_MEM_READ_WRITE);
    }
    // compute densities on available GPUs
    unsigned int j, thread_id;
    unsigned int n_selected_cols = selected_cols.size();
    #pragma omp parallel for default(none)\
                             private(j,thread_id)\
                             firstprivate(n_selected_cols,n_rows,\
                                          n_wg,wgsize)\
                             shared(coords,selected_cols,gpus,\
                                    densities,bandwidths)\
                             num_threads(n_gpus)\
                             schedule(dynamic,1)
    for (j=0; j < n_selected_cols; ++j) {
      thread_id = omp_get_thread_num();
      densities[j] = compute_densities_1d(&gpus[thread_id]
                                        , coords
                                        , n_rows
                                        , j
                                        , bandwidths[j]
                                        , n_wg
                                        , wgsize);
    }
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

    // run computation
    std::vector<std::vector<float>> densities;
    densities = compute_densities(selected_cols
                                , coords
                                , n_rows
                                , bandwidths);

    // output: densities
    for (std::size_t i=0; i < n_rows; ++i) {
      for (std::size_t j=0; j < selected_cols.size(); ++j) {
        Tools::IO::out() << " " << densities[j][i];
      }
      Tools::IO::out() << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Dens'

