
#include "densities_cuda.hpp"
#include "tools.hpp"

#include <cuda.h>

#define WGSIZE 128
#define N_STREAMS 4

namespace {

  //// CUDA kernels

  /* local probability from Epanechnikov kernel */
  __device__ float
  epa(float x
    , float ref_scaled_neg
    , float h_inv) {
    float p = fma(h_inv, x, ref_scaled_neg);
    p *= p;
    if (p <= 1.0f) {
      p = h_inv * fma(p, -0.75f, 0.75f);
    } else {
      p = 0.0f;
    }
    return p;
  }

  /* pre-reduce probabilities inside block */
  __device__ void
  local_prob_reduction(unsigned int tid
                     , float* p_wg
                     , float* P_partial) {
    unsigned int stride;
    unsigned int bid = blockIdx.x;
    // reduce locally
    for (stride=WGSIZE/2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        p_wg[tid] += p_wg[tid+stride];
      }
    }
    if (tid == 0) {
      P_partial[bid] = p_wg[0] / ((float) WGSIZE);
    }
  }

  /* compute partial 1D probabilities */
  __global__ void
  partial_probs_1d(unsigned int offset
                 , const float* sorted_coords
                 , unsigned int n_rows
                 , float* P_partial
                 , float h_inv_1
                 , float ref_scaled_neg_1) {
    __shared__ float p_wg[WGSIZE];
    unsigned int bid = blockIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid*bsize+tid + offset;
    // probability for every frame
    if (gid < n_rows) {
      p_wg[tid] = epa(sorted_coords[gid]
                    , ref_scaled_neg_1
                    , h_inv_1);
    } else {
      p_wg[tid] = 0.0f;
    }
    // pre-reduce inside workgroup
    local_prob_reduction(tid, p_wg, P_partial);
  }

  /* compute partial 2D probabilities */
  __global__ void
  partial_probs_2d(unsigned int offset
                 , const float* sorted_coords
                 , unsigned int n_rows
                 , float* P_partial
                 , float h_inv_1
                 , float ref_scaled_neg_1
                 , float h_inv_2
                 , float ref_scaled_neg_2) {
    __shared__ float p_wg[WGSIZE];
    unsigned int bid = blockIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid*bsize+tid + offset;
    // probability for every frame
    if (gid < n_rows) {
      p_wg[tid] = epa(sorted_coords[gid]
                    , ref_scaled_neg_1
                    , h_inv_1)
                * epa(sorted_coords[n_rows+gid]
                    , ref_scaled_neg_2
                    , h_inv_2);
    } else {
      p_wg[tid] = 0.0f;
    }
    // pre-reduce inside workgroup
    local_prob_reduction(tid, p_wg, P_partial);
  }

  /* compute partial 3D probabilities */
  __global__ void
  partial_probs_3d(unsigned int offset
                 , const float* sorted_coords
                 , unsigned int n_rows
                 , float* P_partial
                 , float h_inv_1
                 , float ref_scaled_neg_1
                 , float h_inv_2
                 , float ref_scaled_neg_2
                 , float h_inv_3
                 , float ref_scaled_neg_3) {
    __shared__ float p_wg[WGSIZE];
    unsigned int bid = blockIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid*bsize+tid + offset;
    // probability for every frame
    if (gid < n_rows) {
      p_wg[tid] = epa(sorted_coords[gid]
                    , ref_scaled_neg_1
                    , h_inv_1)
                * epa(sorted_coords[n_rows+gid]
                    , ref_scaled_neg_2
                    , h_inv_2)
                * epa(sorted_coords[2*n_rows+gid]
                    , ref_scaled_neg_3
                    , h_inv_3);
    } else {
      p_wg[tid] = 0.0f;
    }
    // pre-reduce inside workgroup
    local_prob_reduction(tid, p_wg, P_partial);
  }

  /* reduce partial probabilities */
  __global__ void
  sum_partial_probs_atomic(float* P_partial
                         , float* P
                         , unsigned int i_ref
                         , unsigned int n_partials
                         , unsigned int n_wg) {
    __shared__ float p_wg[WGSIZE];
    unsigned int stride;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int gid = bid*bsize+tid;
    unsigned int gid2 = gid + bsize*gridDim.x;
    // store probs locally for reduction
    if (gid2 < n_partials) {
      // initial double load and first reduction
      p_wg[tid] = P_partial[gid] + P_partial[gid2];
    } else if (gid < n_partials) {
      p_wg[tid] = P_partial[gid];
    } else {
      p_wg[tid] = 0.0f;
    }
    for (stride=WGSIZE/2; stride > 32; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        p_wg[tid] += p_wg[tid+stride];
      }
    }
    // unroll loop inside warp (intrinsic sync!)
    __syncthreads();
    if (tid < 32) {
      p_wg[tid] += p_wg[tid+32];
      p_wg[tid] += p_wg[tid+16];
      p_wg[tid] += p_wg[tid+8];
      p_wg[tid] += p_wg[tid+4];
      p_wg[tid] += p_wg[tid+2];
      p_wg[tid] += p_wg[tid+1];
    }
    if (tid == 0) {
      atomicAdd(&P[i_ref], p_wg[0] / ((float) n_wg));
    }
  }

} // end local namespace


std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<std::size_t> i_cols
                 , std::vector<float> h
                 , std::vector<std::size_t> tau) {
  std::size_t n_dim = i_cols.size();
  if (n_dim < 1 || 3 < n_dim) {
    std::cerr << "error: can only compute combined probabilities in 1, 2 or 3 "
              << "dimensions."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (h.size() != n_dim) {
    std::cerr << "error: number of bandwidth parameters does not match number "
              << "of selected columns."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (tau.size() != n_dim) {
    std::cerr << "error: number of lagtimes (tau) does not match number of "
              << "selected dimensions."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  //TODO include lagtimes 'tau' in coord preparation and densities
  std::vector<float> sorted_coords = Tools::dim1_sorted_coords(coords
                                                             , n_rows
                                                             , i_cols
                                                             , tau);
  // create buffers on device
  unsigned int n_blocks = Tools::min_multiplicator(n_rows
                                                 , WGSIZE);
  unsigned int partial_size = Tools::min_multiplicator(n_blocks
                                                     , WGSIZE) * WGSIZE;
  float* d_sorted_coords;
  cudaMalloc((void**) &d_sorted_coords
           , sizeof(float) * n_blocks*WGSIZE*n_dim);
  float* d_P;
  cudaMalloc((void**) &d_P
           , sizeof(float) * n_rows);
  // one partial result array per stream
  float* d_P_partial[N_STREAMS];
  // copy coords to device
  cudaMemcpy(d_sorted_coords
           , sorted_coords.data()
           , sizeof(float) * n_rows * n_dim
           , cudaMemcpyHostToDevice);
  // initialize P with zeros
  cudaMemset(d_P, 0, sizeof(float) * n_rows);
  // box limits for pruning
  std::vector<float> blimits = Tools::boxlimits(sorted_coords
                                              , WGSIZE
                                              , n_dim);
  // create multiple streams for parallel execution
  cudaStream_t streams[N_STREAMS];
  for (unsigned int i=0; i < N_STREAMS; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc((void**) &d_P_partial[i]
             , sizeof(float) * partial_size);
  }
  // run computation
  std::vector<float> h_inv(n_dim);
  for (std::size_t n=0; n < n_dim; ++n) {
    h_inv[n] = 1.0f/h[n];
  }
  for (unsigned int i=0; i < n_rows; ++i) {
    // set reference
    float ref_1 = coords[i_cols[0]*n_rows + i];
    std::vector<float> ref_scaled_neg(n_dim);
    for (std::size_t n=0; n < n_dim; ++n) {
      ref_scaled_neg[n] = -h_inv[n] * coords[i_cols[n]*n_rows + i];
    }
    // pruning on 1st dimension
    auto boxes_from_to = Tools::min_max_box(blimits
                                          , ref_1
                                          , h[0]);
    unsigned int bfrom = boxes_from_to.first;
    unsigned int bto = boxes_from_to.second;
    unsigned int n_blocks_pruned = bto - bfrom + 1;
    unsigned int offset = bfrom * WGSIZE;
    unsigned int rng = Tools::min_multiplicator(n_blocks_pruned
                                              , WGSIZE)
                     * WGSIZE;
    unsigned int i_stream = i % N_STREAMS;
    // compute partials asynchronously on different streams
    switch(n_dim) {
    case 1:
      partial_probs_1d <<< n_blocks_pruned
                         , WGSIZE
                         , 0
                         , streams[i_stream] >>> (offset
                                                , d_sorted_coords
                                                , n_rows
                                                , d_P_partial[i_stream]
                                                , h_inv[0]
                                                , ref_scaled_neg[0]);
      break;
    case 2:
      partial_probs_2d <<< n_blocks_pruned
                         , WGSIZE
                         , 0
                         , streams[i_stream] >>> (offset
                                                , d_sorted_coords
                                                , n_rows
                                                , d_P_partial[i_stream]
                                                , h_inv[0]
                                                , ref_scaled_neg[0]
                                                , h_inv[1]
                                                , ref_scaled_neg[1]);
      break;
    case 3:
      partial_probs_3d <<< n_blocks_pruned
                         , WGSIZE
                         , 0
                         , streams[i_stream] >>> (offset
                                                , d_sorted_coords
                                                , n_rows
                                                , d_P_partial[i_stream]
                                                , h_inv[0]
                                                , ref_scaled_neg[0]
                                                , h_inv[1]
                                                , ref_scaled_neg[1]
                                                , h_inv[2]
                                                , ref_scaled_neg[2]);
      break;
    default:
      // this should never happen
      exit(EXIT_FAILURE);
    }
    // compute P(i) from partials
    sum_partial_probs_atomic <<< rng/2
                               , WGSIZE
                               , 0
                               , streams[i_stream] >>> (d_P_partial[i_stream]
                                                      , d_P
                                                      , i
                                                      , n_blocks_pruned
                                                      , n_blocks);
  }
  // get results from GPU
  std::vector<float> P(n_rows);
  // sync over streams
  cudaThreadSynchronize();
  cudaMemcpy(P.data()
           , d_P
           , sizeof(float) * n_rows
           , cudaMemcpyDeviceToHost);
  // free the mallocs!
  cudaFree(d_sorted_coords);
  cudaFree(d_P);
  for (unsigned int i=0; i < N_STREAMS; ++i) {
    cudaFree(d_P_partial);
  }
  return P;
}

