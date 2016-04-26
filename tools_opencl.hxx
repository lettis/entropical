
#include <iostream>
#include "tools_opencl.hpp"

namespace Tools {
namespace OCL {

  template <typename NUM> void
  set_kernel_arg(GPUElement* gpu
               , std::string kernel
               , unsigned int n_param
               , NUM kernel_arg) {
    std::cerr << "setting: " << kernel << " " << n_param << ": " << kernel_arg << std::endl;
    check_error(clSetKernelArg(gpu->kernels[kernel]
                             , n_param
                             , sizeof(NUM)
                             , &kernel_arg)
              , "clSetKernelArg");
  }

} // end namespace Tools::OCL
} // end namespace Tools

