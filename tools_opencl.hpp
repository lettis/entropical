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
  std::string err_to_string(int err);

} // end namespace Tools::OCL
} // end namespace Tools

