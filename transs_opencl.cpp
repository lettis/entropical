
#include "transs_opencl.hpp"

#include <vector>
#include <fstream>
#include <streambuf>

namespace Transs {
namespace OCL {

  std::string
  load_kernel_source(std::string fname) {
    std::ifstream fin(fname);
    std::string src;
    fin.seekg(0, std::ios::end);
    src.reserve(fin.tellg());
    fin.seekg(0, std::ios::beg);
    src.assign((std::istreambuf_iterator<char>(fin)),std::istreambuf_iterator<char>());
    return src;
  }

  std::pair<cl_uint, cl_platform_id>
  gpus() {
    cl_uint n_devices;
    cl_uint n_platforms;
    clGetPlatformIDs(0, NULL, &n_platforms);
    std::vector<cl_platform_id> platforms(n_platforms);
    clGetPlatformIDs(n_platforms, platforms.data(), NULL);
    for (cl_platform_id p: platforms) {
      clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
      if (n_devices > 0) {
        return {n_devices, p};
      }
    }
    return {0, 0};
  }

  void
  setup_gpu(GPUElement& gpu
          , cl_platform_id platform
          , std::string kernel_src) {
    //TODO
  }

  std::pair<float, float>
  transfer_entropies(GPUElement& gpu
                   , std::size_t x
                   , std::size_t y
                   , const float* coords
                   , std::size_t n_rows
                   , std::size_t n_cols
                   , const std::vector<float> bandwidths
                   , unsigned int wgsize) {
    //TODO
  }

} // end namespace Transs::OCL
} // end namespace Transs


