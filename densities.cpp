
#include "densities.hpp"

#include <algorithm>

namespace {

  std::vector<float>
  prepare_coords(Tools::OCL::GPUElement* gpu
               , std::string bufname
               , const float* coords
               , std::size_t n_rows
               , std::size_t i_col
               , std::size_t wgsize) {
    using Tools::OCL::check_error;
    // pre-sort coordinates
    std::vector<float> sorted_coords(n_rows);
    for (std::size_t i=0; i < n_rows; ++i) {
      sorted_coords[i] = coords[i_col*n_rows + i];
    }
    std::sort(sorted_coords.begin(), sorted_coords.end());
    // copy (sorted) coords of given column to device
    check_error(clEnqueueWriteBuffer(gpu->q
                                   , gpu->buffers[bufname]
                                   , CL_TRUE
                                   , 0
                                   , sizeof(float) * n_rows
                                   , sorted_coords.data()
                                   , 0
                                   , NULL
                                   , NULL)
              , "clEnqueueWriteBuffer");
    check_error(clFlush(gpu->q), "clFlush");
    // limits for box-assisted pruning
    return Tools::boxlimits(sorted_coords, wgsize);
  }
  
  void
  stagewise_reduction(Tools::OCL::GPUElement* gpu
                    , unsigned int i_ref
                    , unsigned int n_partials
                    , unsigned int n_wg
                    , std::size_t wgsize) {
    std::string partial_buf1 = "P_partial";
    std::string partial_buf2 = "P_partial_reduct";
    auto set_partial_bufs = [&] () -> void {
      Tools::OCL::set_kernel_buf_arg(gpu
                                   , "sum_partial_probs"
                                   , 0
                                   , partial_buf1);
      Tools::OCL::set_kernel_buf_arg(gpu
                                   , "sum_partial_probs"
                                   , 5
                                   , partial_buf2);
    };
    Tools::OCL::set_kernel_buf_arg(gpu
                                 , "sum_partial_probs"
                                 , 1
                                 , "P");
    Tools::OCL::set_kernel_scalar_arg(gpu
                                    , "sum_partial_probs"
                                    , 2 // i_ref
                                    , i_ref);
    Tools::OCL::set_kernel_scalar_arg(gpu
                                    , "sum_partial_probs"
                                    , 3 // n_partials
                                    , (unsigned int) n_partials);
    Tools::OCL::set_kernel_scalar_arg(gpu
                                    , "sum_partial_probs"
                                    , 4 // n_wg
                                    , (unsigned int) n_wg);
    set_partial_bufs();
    unsigned int rng = Tools::min_multiplicator(n_partials, wgsize) * wgsize;
    while (rng > 1) {
      // reduce on current stage
      if (rng < wgsize) {
        gpu->nq_range_offset("sum_partial_probs"
                           , 0
                           , wgsize
                           , wgsize);
      } else {
        gpu->nq_range_offset("sum_partial_probs"
                           , 0
                           , rng
                           , wgsize);
      }
      // next stage with swapped partial buffers for correct reduction
      rng /= wgsize;
      Tools::OCL::set_kernel_scalar_arg(gpu
                                      , "sum_partial_probs"
                                      , 3
                                      , rng);
      std::swap(partial_buf1, partial_buf2);
      set_partial_bufs();
    }
    Tools::OCL::check_error(clFlush(gpu->q), "clFlush");
  }

} // end local namespace



void
prepare_gpus_1d(std::vector<Tools::OCL::GPUElement>& gpus
              , unsigned int wgsize1d
              , unsigned int n_wg
              , std::size_t n_rows
              , unsigned int partial_size) {
  //TODO embed kernel source in header
  std::string kernel_src = Tools::OCL::load_kernel_source("kernels.cl");
  for (unsigned int i=0; i < gpus.size(); ++i) {
    Tools::OCL::setup_gpu(&gpus[i]
                        , kernel_src
                        , {"partial_probs_1d"
                         , "sum_partial_probs"}
                        , wgsize1d
                        , 0
                        , 0);
    Tools::OCL::create_buffer(&gpus[i]
                            , "sorted_coords"
                            , sizeof(float) * n_wg * wgsize1d
                            , CL_MEM_READ_ONLY);
    Tools::OCL::create_buffer(&gpus[i]
                            , "P"
                            , sizeof(float) * n_rows
                            , CL_MEM_WRITE_ONLY);
    Tools::OCL::create_buffer(&gpus[i]
                            , "P_partial"
                            , sizeof(float) * partial_size
                            , CL_MEM_READ_WRITE);
    Tools::OCL::create_buffer(&gpus[i]
                            , "P_partial_reduct"
                            , sizeof(float) * partial_size / wgsize1d
                            , CL_MEM_READ_WRITE);
  }
}

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
  // transmit sorted coords to GPU and separate into boxes
  std::vector<float> boxlimits = prepare_coords(gpu
                                              , "sorted_coords"
                                              , coords
                                              , n_rows
                                              , i_col
                                              , wgsize);
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
    gpu->nq_range_offset("partial_probs_1d"
                       , min_max.first * wgsize
                       , mm_range * wgsize
                       , wgsize);
    // compute P(i) from partials
    stagewise_reduction(gpu, i, mm_range, n_wg, wgsize);
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

std::vector<float>
compute_densities_2d(Tools::OCL::GPUElement* gpu
                   , const float* coords
                   , std::size_t n_rows
                   , std::size_t i_col[2]
                   , float h[2]
                   , std::size_t n_wg[2]
                   , std::size_t wgsize) {
  //TODO implement
}

