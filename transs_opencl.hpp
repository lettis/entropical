#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <utility>
#include <vector>
#include <map>
#include <string>

namespace Transs {
namespace OCL {

  /**
   * Compute transfer entropies x -> y and y -> x
   * for column x and y of coords on the specified GPU.
   */
//  std::pair<float, float>
//  transfer_entropies(GPUElement& gpu
//                   , std::size_t x
//                   , std::size_t y
//                   , const float* coords
//                   , unsigned int n_rows
//                   , unsigned int tau
//                   , const std::vector<float> bandwidths
//                   , unsigned int wgsize
//                   , unsigned int n_workgroups);

} // end namespace Transs::OCL
} // end namespace Transs

