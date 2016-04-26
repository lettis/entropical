#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <map>
#include <vector>

namespace Tools {
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
   * Translate OpenCL error code to human readable string.
   * Based on OpenCL 1.2 header.
   */
  std::string
  err_to_string(int err);

  /**
   * Print error message to std::cerr to tell the user
   * that no GPUs have been found.
   */
  void
  print_no_gpus_errmsg();


  /**
   * Compute maximally available workgroup size for given
   * GPU and needed kernel resources (will be a multiple of 64).
   */
  unsigned int
  max_wgsize(GPUElement& gpu
           , unsigned int bytes_per_workitem);

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
   * Creates context, queue and compiled kernels for GPU.
   */
  void
  setup_gpu(GPUElement& gpu
          , std::string kernel_src
          , std::vector<std::string> used_kernels
          , unsigned int wgsize);

  /**
   * Creates buffer on given GPU device.
   */
  void
  create_buffer(GPUElement& gpu
              , std::string bname
              , std::size_t bsize
              , cl_mem_flags bflags);

  /**
   * Release OpenCL resources for given GPU.
   */
  void
  cleanup_gpu(GPUElement& gpu);

  /**
   * Templated helper function to set kernel argument.
   * Will exit and provide with error message if not successful.
   *
   * valid types:
   *   string:       interpreted as name of a buffer
   *   unsigned int: interpreted as scalar value
   *   float:        interpreted as scalar value
   */
  template <typename T> void
  set_kernel_arg(GPUElement& gpu
               , std::string kernel
               , unsigned int n_param
               , T kernel_arg);

} // end namespace Tools::OCL
} // end namespace Tools

#include "tools_opencl.hxx"

