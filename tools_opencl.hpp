#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

namespace Tools {
namespace OCL {

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
   * GPU and needed kernel resources.
   */
  unsigned int
  max_wgsize(GPUElement& gpu
           , unsigned int needed_kernel_resources);

} // end namespace Tools::OCL
} // end namespace Tools

