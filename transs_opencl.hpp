#pragma once

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
   * Represents GPU computing element,
   * i.e. a single GPU with context, queue,
   * compiled kernels and buffers.
   */
  struct GPUElement {
    cl_device_id i_dev;
    cl_platform_id i_platform;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    std::map<std::string, cl_kernel> kernels;
    std::map<std::string, cl_mem> buffers;
  };

  /**
   * @returns Compute elements for available GPUs
   */
  std::vector<GPUElement>
  gpus();

  /**
   * @returns OpenCL kernel source directly from file 'fname'
   */
  std::string
  load_kernel_source(std::string fname);

  /**
   * Creates context, queue and compiled kernels for GPU and
   * allocates buffers needed for transfer entropy computation.
   */
  void
  setup_gpu(GPUElement& gpu
          , std::string kernel_src
          , unsigned int wgsize
          , unsigned int n_workgroups
          , unsigned int n_rows);

  /**
   * Release OpenCL resources for given GPU.
   */
  void
  cleanup_gpu(GPUElement& gpu);

  /**
   * Callback function for OpenCL errors.
   */
  void
  pfn_notify(const char* errinfo
           , const void* private_info
           , size_t cb
           , void* user_data);

  /**
   * Check if OpenCL was without error.
   * If not, give a proper error message and abort.
   */
  void
  check_error(cl_int err_code
            , const char* err_name);

  /**
   * Compute transfer entropies x -> y and y -> x
   * for column x and y of coords on the specified GPU.
   */
  std::pair<float, float>
  transfer_entropies(GPUElement& gpu
                   , std::size_t x
                   , std::size_t y
                   , const float* coords
                   , unsigned int n_rows
                   , unsigned int tau
                   , const std::vector<float> bandwidths
                   , unsigned int wgsize
                   , unsigned int n_workgroups);

} // end namespace Transs::OCL
} // end namespace Transs

