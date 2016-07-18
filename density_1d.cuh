#pragma once
/*
Copyright (c) 2016, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "tools.cuh"
#include <omp.h>


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



template <typename type_in
        , typename type_out>
__device__ type_out
density_1d_interaction(type_in* cache
                     , unsigned int n_cols
                     , unsigned int i_ref
                     , unsigned int i_cache) {
 //TODO: implement me!
}

////////////////////////////////////////////////////////////

template <typename type_in
        , typename type_out>
__global__ void
density_1d_krnl(unsigned int i_offset
              , unsigned int i_from
              , unsigned int i_to
              , type_in* coords
              , unsigned int n_rows
              , unsigned int n_cols
              , type_out* results) {
  extern __shared__ type_in smem[];
  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int bsize = blockDim.x;
  unsigned int gid = bid * bsize + tid + i_from;
  // load coords for density_1d interaction computation
  // into shared memory
  int n_cache_rows = min(bsize, n_rows-i_offset);
  if (tid < n_cache_rows) {
    for (unsigned int j=0; j < n_cols; ++j) {
      smem[tid*n_cols+j] = coords[(tid+i_offset)*n_cols+j];
    }
  }
  __syncthreads();
  // compute density_1d interaction
  if (gid < i_to) {
    unsigned int i_ref = tid + bsize;
    // load reference for re-use into shared memory
    for (unsigned int j=0; j < n_cols; ++j) {
      smem[i_ref*n_cols+j] = coords[gid*n_cols+j];
    }
    // accumulate density_1d interactions between
    // reference and cached coordinates
    type_out acc = 0;
    for (unsigned int i=0; i < n_cache_rows; ++i) {
      acc += density_1d_interaction(smem
                                  , n_cols
                                  , i_ref
                                  , i);
    }
    results[gid] += acc;
  }
}

//// driver functions

template <typename type_in
        , typename type_out
        , unsigned int blocksize>
std::vector<type_out>
density_1d_per_gpu(const type_in* coords
                 , unsigned int n_rows
                 , unsigned int n_cols
                 , unsigned int i_from
                 , unsigned int i_to
                 , unsigned int i_gpu) {
  // GPU setup
  cudaSetDevice(i_gpu);
  type_in* d_coords;
  type_out* d_results;
  cudaMalloc((void**) &d_coords
           , sizeof(type_in) * n_rows * n_cols);
  cudaMalloc((void**) &d_results
           , sizeof(type_out) * n_rows);
  cudaMemset(d_results
           , 0
           , sizeof(type_out) * n_rows);
  cudaMemcpy(d_coords
           , coords
           , sizeof(type_in) * n_rows * n_cols
           , cudaMemcpyHostToDevice);
  // determine memory size and block dimensions
  int max_shared_mem;
  cudaDeviceGetAttribute(&max_shared_mem
                       , cudaDevAttrMaxSharedMemoryPerBlock
                       , i_gpu);
  check_error("getting info: max. shared memory on gpu");
  unsigned int shared_mem = 2 * blocksize * n_cols * sizeof(type_in);
  if (shared_mem > max_shared_mem) {
    std::cerr << "error: max. shared mem per block too small on this GPU.\n"
              << "       either reduze blocksize or get a better GPU."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned int blockrange = (i_to-i_from) / blocksize;
  if (blockrange*blocksize < (i_to-i_from)) {
    ++blockrange;
  }
  // run computation
  for (unsigned int i=0; i*blocksize < n_rows; ++i) {
    density_1d_krnl <<< blockrange
                      , blocksize
                      , shared_mem >>> (i*blocksize
                                      , i_from
                                      , i_to
                                      , d_coords
                                      , n_rows
                                      , n_cols
                                      , d_results);
  }
  cudaDeviceSynchronize();
  check_error("kernel loop");
  // retrieve results from GPU
  std::vector<type_out> results(n_rows);
  cudaMemcpy(results.data()
           , d_results
           , sizeof(type_out) * n_rows
           , cudaMemcpyDeviceToHost);
  // GPU cleanup
  cudaFree(d_coords);
  cudaFree(d_results);
  return results;
}

template <typename type_out>
std::vector<type_out> 
density_1d_reduce(std::vector<type_out>& partial_results) {
  unsigned int n_rows = partial_results[0].size();
  std::vector<type_out> results(n_rows);
  for (std::vector<type_out>& part: partial_results) {
    for (unsigned int i; i < n_rows; ++i) {
      results[i] += part[i];
    }
  }
  return results;
}

template <typename type_in
        , typename type_out
        , unsigned int blocksize>
std::vector<type_out>
density_1d(const type_in* coords
         , unsigned int n_rows
         , unsigned int n_cols) {
  int n_gpus;
  cudaGetDeviceCount(&n_gpus);
  if (n_gpus == 0) {
    std::cerr << "error: nu CUDA-compatible GPUs found" << std::endl;
    exit(EXIT_FAILURE);
  }
  int gpurange = n_rows / n_gpus;
  //TODO: check gpurange against n_rows & max blocksize,
  //      if too large: split further into smaller chunks
  int i_gpu;
  std::vector<type_out> partial_results(n_gpus);
  #pragma omp parallel for default(none)\
    private(i_gpu)\
    firstprivate(n_gpus,n_rows,n_cols,gpurange)\
    shared(partial_results,coords)\
    num_threads(n_gpus)\
    schedule(dynamice,1)
  for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
    partial_results[i_gpu] = density_1d_per_gpu(coords
                                              , n_rows
                                              , n_cols
                                              , i_gpu*gpurange
                                              , i_gpu == (n_gpus-1)
                                                  ? n_rows
                                                  : (i_gpu+1)*gpurange
                                              , i_gpu);
  }
  return density_1d_reduce(partial_results);
}

