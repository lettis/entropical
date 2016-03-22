
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
    src.assign((std::istreambuf_iterator<char>(fin))
             , std::istreambuf_iterator<char>());
    return src;
  }

  std::vector<GPUElement>
  gpus() {
    std::vector<GPUElement> gpus;
    cl_uint n_devices;
    cl_uint n_platforms;
    check_error(clGetPlatformIDs(0
                               , NULL
                               , &n_platforms));
    std::vector<cl_platform_id> platforms(n_platforms);
    check_error(clGetPlatformIDs(n_platforms
                               , platforms.data()
                               , NULL), "clGetPlatformIDs");
    for (cl_platform_id p: platforms) {
      check_error(clGetDeviceIDs(p
                               , CL_DEVICE_TYPE_GPU
                               , 0
                               , NULL
                               , &n_devices), "clGetDeviceIDs");
      if (n_devices > 0) {
        std::vector<cl_device_id> dev_ids(n_devices);
        check_error(clGetDeviceIDs(p
                                 , CL_DEVICE_TYPE_GPU
                                 , n_devices
                                 , dev_ids.data()
                                 , NULL), "clGetDeviceIDs");
        gpus.resize(n_devices);
        for (cl_uint i=0; i < n_devices; ++i) {
          gpus[i].i_dev = dev_ids[i];
          gpus[i].i_platform = p;
        }
      }
    }
    return gpus;
  }

  void
  setup_gpu(GPUElement& gpu
          , std::string kernel_src
          , unsigned int wgsize) {
    // prepare kernel source
    kernel_src = std::string("#define WGSIZE ")
                 + std::to_string(wgsize)
                 + std::string("\n")
                 + kernel_src;
    // create context
    cl_int err;
    gpu.ctx = clCreateContext(NULL
                            , 1
                            , &gpu.i_dev
                            , &pfn_notify
                            , NULL
                            , &err);
    check_error(err, "clCreateContex");
    // create program
    gpu.prog = clCreateProgramWithSource(gpu.ctx
                                       , 1
                                       , &kernel_src.c_str()
                                       , NULL
                                       , &err);
    check_error(err, "clCreateProgramWithSource");
    // compile kernels
    if (clBuildProgram(gpu.prog
                     , 1
                     , &gpu.i_dev
                     , ""
                     , NULL
                     , NULL) != CL_SUCCESS) {
      char buffer[10240];
      clGetProgramBuildInfo(gpu.prog
                          , gpu.i_dev
                          , CL_PROGRAM_BUILD_LOG
                          , sizeof(buffer)
                          , buffer
                          , NULL);
      std::cerr << "CL Compilation failed:" << std::endl
                << buffer << std::endl;
      abort();
    }
    //TODO create kernels and buffers
  }

  void
  pfn_notify(const char *errinfo
           , const void *private_info
           , size_t cb
           , void *user_data) {
	  std::cerr << "OpenCL Error (via pfn_notify): " << errinfo << std::endl;
  }

  void
  check_error(cl_int err_code, const char* err_name) {
    if (err_code != CL_SUCCESS) {
      std::cerr << "error:  " << err_name << ": " << err_code << std::endl;
      exit(EXIT_FAILURE);
    }
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


