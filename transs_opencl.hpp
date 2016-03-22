#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <utility>

namespace Transs {
namespace OCL {

  struct GPUElement {
    cl_device_id gpu;
    cl_context ctx;
    cl_command_queue queue;
    cl_program prog;
    std::vector<cl_kernel> kernels;
    std::vector<cl_mem> buffers;
  };

  std::string load_kernel_source(std::string fname);

  void setup_gpu(GPUElement& gpu, cl_platform_id platform, std::string kernel_src);

  std::pair<float, float> transfer_entropies(GPUElement& gpu
                                           , std::size_t x
                                           , std::size_t y
                                           , const float* coords
                                           , std::size_t n_rows
                                           , std::size_t n_cols
                                           , const std::vector<float> bandwidths
                                           , unsigned int wgsize);

  /**
   * @returns number of available GPUs and their platform id
   */
  std::pair<cl_uint, cl_platform_id> gpus();



} // end namespace Transs::OCL
} // end namespace Transs

