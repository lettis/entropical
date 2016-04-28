
#include <iostream>
#include "tools_opencl.hpp"

namespace Tools {
namespace OCL {

  template <typename NUM> void
  set_kernel_scalar_arg(GPUElement* gpu
                      , std::string kernel
                      , unsigned int n_param
                      , NUM kernel_arg) {
    check_error(clSetKernelArg(gpu->kernels[kernel]
                             , n_param
                             , sizeof(NUM)
                             , &kernel_arg)
              , "clSetKernelArg");
  }

} // end namespace Tools::OCL
} // end namespace Tools

