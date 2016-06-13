
#include "densities_opencl.hpp"

//#define NDEBUG
#include <assert.h>

#include <algorithm>

namespace {

  /**
   * Copy coords to GPU and return box-separation.
   */
  std::vector<float>
  prepare_coords(Tools::OCL::GPUElement* gpu
               , const float* coords
               , std::size_t n_rows
               , std::vector<std::size_t> col_indices
               , std::size_t wgsize) {
    using Tools::OCL::check_error;
    unsigned int n_dim = col_indices.size();
    // pre-sort coordinates
    std::vector<float> sorted_coords = Tools::dim1_sorted_coords(coords
                                                               , n_rows
                                                               , col_indices);
    // copy (sorted) coords to device
    check_error(clEnqueueWriteBuffer(gpu->q
                                   , gpu->buffers["sorted_coords"]
                                   , CL_TRUE
                                   , 0
                                   , sizeof(float) * n_rows * n_dim
                                   , sorted_coords.data()
                                   , 0
                                   , NULL
                                   , NULL)
              , "clEnqueueWriteBuffer");
    // limits for box-assisted pruning
    return Tools::boxlimits(sorted_coords, wgsize, n_dim);
  }
  
  /**
   * Perform stagewise parallel reduction of probabilities directly on GPU.
   */
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



unsigned int
prepare_gpus(std::vector<Tools::OCL::GPUElement>& gpus
           , unsigned int wgsize
           , std::size_t n_rows
           , std::size_t n_dim) {
  assert(1 <= n_dim && n_dim <= 3);
  unsigned int n_wg = Tools::min_multiplicator(n_rows, wgsize);
  unsigned int partial_size = Tools::min_multiplicator(n_wg
                                                     , wgsize) * wgsize;
  //TODO embed kernel source in header
  std::string kernel_src = Tools::OCL::load_kernel_source("kernels.cl");
  for (unsigned int i=0; i < gpus.size(); ++i) {
    Tools::OCL::setup_gpu(&gpus[i]
                        , kernel_src
                        , {"partial_probs_" + std::to_string(n_dim) + "d"
                         , "sum_partial_probs"}
                        , wgsize);
    Tools::OCL::create_buffer(&gpus[i]
                            , "sorted_coords"
                            , sizeof(float) * n_wg * wgsize * n_dim
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
                            , sizeof(float) * partial_size / wgsize
                            , CL_MEM_READ_WRITE);
  }
  return n_wg;
}


std::vector<float>
combined_densities(Tools::OCL::GPUElement* gpu
                 , const float* coords
                 , std::size_t n_rows
                 , std::vector<std::size_t> i_cols
                 , std::vector<float> hs
                 , std::size_t n_wg
                 , std::size_t wgsize) {
  assert(i_cols.size() == hs.size()
      && 1 <= i_cols.size()
      && i_cols.size() <= 3);
  using Tools::OCL::check_error;
  unsigned int n_dim = i_cols.size();
  std::string kernel_name = "partial_probs_" + std::to_string(n_dim) + "d";
  // transmit sorted coords to GPU and separate into boxes
  std::vector<float> boxlimits = prepare_coords(gpu
                                              , coords
                                              , n_rows
                                              , i_cols
                                              , wgsize);
  // run kernel-loop over all frames
  Tools::OCL::set_kernel_buf_arg(gpu
                               , kernel_name
                               , 0
                               , "sorted_coords");
  Tools::OCL::set_kernel_scalar_arg(gpu
                                  , kernel_name
                                  , 1
                                  , (unsigned int) n_rows);
  Tools::OCL::set_kernel_buf_arg(gpu
                               , kernel_name
                               , 2
                               , "P_partial");

  float h_inv_1=0, h_inv_2=0, h_inv_3=0;
  h_inv_1 = 1.0f/hs[0];
  Tools::OCL::set_kernel_scalar_arg(gpu
                                  , kernel_name
                                  , 3
                                  , h_inv_1);
  if (n_dim > 1) {
    h_inv_2 = 1.0f/hs[1];
    Tools::OCL::set_kernel_scalar_arg(gpu
                                    , kernel_name
                                    , 5
                                    , h_inv_2);
    if (n_dim == 3) {
      h_inv_3 = 1.0f/hs[2];
      Tools::OCL::set_kernel_scalar_arg(gpu
                                      , kernel_name
                                      , 7
                                      , h_inv_3);
    }
  }
  for (unsigned int i=0; i < n_rows; ++i) {
    // prune full range to limit of bandwidth
    float ref_val = coords[i_cols[0]*n_rows + i];
    auto min_max = Tools::min_max_box(boxlimits, ref_val, hs[0]);
    std::size_t mm_range = min_max.second - min_max.first + 1;
    // set reference
    float ref_scaled_neg = -h_inv_1 * ref_val;
    Tools::OCL::set_kernel_scalar_arg(gpu
                                    , kernel_name
                                    , 4
                                    , ref_scaled_neg);
    if (n_dim > 1) {
      ref_scaled_neg = -h_inv_2 * coords[i_cols[1]*n_rows + i];
      Tools::OCL::set_kernel_scalar_arg(gpu
                                      , kernel_name
                                      , 6
                                      , ref_scaled_neg);
      if (n_dim == 3) {
        ref_scaled_neg = -h_inv_3 * coords[i_cols[2]*n_rows + i];
        Tools::OCL::set_kernel_scalar_arg(gpu
                                        , kernel_name
                                        , 8
                                        , ref_scaled_neg);
      }
    }
    // compute partials in workgroups
    gpu->nq_range_offset(kernel_name
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
