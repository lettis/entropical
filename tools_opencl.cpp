
#include <iostream>

#include "tools_opencl.hpp"


namespace Tools {
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
                               , &n_platforms), "clGetPlatformIDs");
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

  std::string err_to_string(int err) {
    auto format = [](int err, std::string name) {
      return std::string(name)
              + std::string(" (")
              + std::to_string(err)
              + std::string(")");
    };
    switch(err) {
      case CL_SUCCESS:
        return format(err, "CL_SUCCESS");
      case CL_DEVICE_NOT_FOUND:
        return format(err, "CL_DEVICE_NOT_FOUND");
      case CL_DEVICE_NOT_AVAILABLE:
        return format(err, "CL_DEVICE_NOT_AVAILABLE");
      case CL_COMPILER_NOT_AVAILABLE:
        return format(err, "CL_COMPILER_NOT_AVAILABLE");
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return format(err, "CL_MEM_OBJECT_ALLOCATION_FAILURE");
      case CL_OUT_OF_RESOURCES:
        return format(err, "CL_OUT_OF_RESOURCES");
      case CL_OUT_OF_HOST_MEMORY:
        return format(err, "CL_OUT_OF_HOST_MEMORY");
      case CL_PROFILING_INFO_NOT_AVAILABLE:
        return format(err, "CL_PROFILING_INFO_NOT_AVAILABLE");
      case CL_MEM_COPY_OVERLAP:
        return format(err, "CL_MEM_COPY_OVERLAP");
      case CL_IMAGE_FORMAT_MISMATCH:
        return format(err, "CL_IMAGE_FORMAT_MISMATCH");
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return format(err, "CL_IMAGE_FORMAT_NOT_SUPPORTED");
      case CL_BUILD_PROGRAM_FAILURE:
        return format(err, "CL_BUILD_PROGRAM_FAILURE");
      case CL_MAP_FAILURE:
        return format(err, "CL_MAP_FAILURE");
      case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        return format(err, "CL_MISALIGNED_SUB_BUFFER_OFFSET");
      case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return format(err, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
      case CL_COMPILE_PROGRAM_FAILURE:
        return format(err, "CL_COMPILE_PROGRAM_FAILURE");
      case CL_LINKER_NOT_AVAILABLE:
        return format(err, "CL_LINKER_NOT_AVAILABLE");
      case CL_LINK_PROGRAM_FAILURE:
        return format(err, "CL_LINK_PROGRAM_FAILURE");
      case CL_DEVICE_PARTITION_FAILED:
        return format(err, "CL_DEVICE_PARTITION_FAILED");
      case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return format(err, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE");

      case CL_INVALID_VALUE:
        return format(err, "CL_INVALID_VALUE");
      case CL_INVALID_DEVICE_TYPE:
        return format(err, "CL_INVALID_DEVICE_TYPE");
      case CL_INVALID_PLATFORM:
        return format(err, "CL_INVALID_PLATFORM");
      case CL_INVALID_DEVICE:
        return format(err, "CL_INVALID_DEVICE");
      case CL_INVALID_CONTEXT:
        return format(err, "CL_INVALID_CONTEXT");
      case CL_INVALID_QUEUE_PROPERTIES:
        return format(err, "CL_INVALID_QUEUE_PROPERTIES");
      case CL_INVALID_COMMAND_QUEUE:
        return format(err, "CL_INVALID_COMMAND_QUEUE");
      case CL_INVALID_HOST_PTR:
        return format(err, "CL_INVALID_HOST_PTR");
      case CL_INVALID_MEM_OBJECT:
        return format(err, "CL_INVALID_MEM_OBJECT");
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return format(err, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");
      case CL_INVALID_IMAGE_SIZE:
        return format(err, "CL_INVALID_IMAGE_SIZE");
      case CL_INVALID_SAMPLER:
        return format(err, "CL_INVALID_SAMPLER");
      case CL_INVALID_BINARY:
        return format(err, "CL_INVALID_BINARY");
      case CL_INVALID_BUILD_OPTIONS:
        return format(err, "CL_INVALID_BUILD_OPTIONS");
      case CL_INVALID_PROGRAM:
        return format(err, "CL_INVALID_PROGRAM");
      case CL_INVALID_PROGRAM_EXECUTABLE:
        return format(err, "CL_INVALID_PROGRAM_EXECUTABLE");
      case CL_INVALID_KERNEL_NAME:
        return format(err, "CL_INVALID_KERNEL_NAME");
      case CL_INVALID_KERNEL_DEFINITION:
        return format(err, "CL_INVALID_KERNEL_DEFINITION");
      case CL_INVALID_KERNEL:
        return format(err, "CL_INVALID_KERNEL");
      case CL_INVALID_ARG_INDEX:
        return format(err, "CL_INVALID_ARG_INDEX");
      case CL_INVALID_ARG_VALUE:
        return format(err, "CL_INVALID_ARG_VALUE");
      case CL_INVALID_ARG_SIZE:
        return format(err, "CL_INVALID_ARG_SIZE");
      case CL_INVALID_KERNEL_ARGS:
        return format(err, "CL_INVALID_KERNEL_ARGS");
      case CL_INVALID_WORK_DIMENSION:
        return format(err, "CL_INVALID_WORK_DIMENSION");
      case CL_INVALID_WORK_GROUP_SIZE:
        return format(err, "CL_INVALID_WORK_GROUP_SIZE");
      case CL_INVALID_WORK_ITEM_SIZE:
        return format(err, "CL_INVALID_WORK_ITEM_SIZE");
      case CL_INVALID_GLOBAL_OFFSET:
        return format(err, "CL_INVALID_GLOBAL_OFFSET");
      case CL_INVALID_EVENT_WAIT_LIST:
        return format(err, "CL_INVALID_EVENT_WAIT_LIST");
      case CL_INVALID_EVENT:
        return format(err, "CL_INVALID_EVENT");
      case CL_INVALID_OPERATION:
        return format(err, "CL_INVALID_OPERATION");
      case CL_INVALID_GL_OBJECT:
        return format(err, "CL_INVALID_GL_OBJECT");
      case CL_INVALID_BUFFER_SIZE:
        return format(err, "CL_INVALID_BUFFER_SIZE");
      case CL_INVALID_MIP_LEVEL:
        return format(err, "CL_INVALID_MIP_LEVEL");
      case CL_INVALID_GLOBAL_WORK_SIZE:
        return format(err, "CL_INVALID_GLOBAL_WORK_SIZE");
      case CL_INVALID_PROPERTY:
        return format(err, "CL_INVALID_PROPERTY");
      case CL_INVALID_IMAGE_DESCRIPTOR:
        return format(err, "CL_INVALID_IMAGE_DESCRIPTOR");
      case CL_INVALID_COMPILER_OPTIONS:
        return format(err, "CL_INVALID_COMPILER_OPTIONS");
      case CL_INVALID_LINKER_OPTIONS:
        return format(err, "CL_INVALID_LINKER_OPTIONS");
      case CL_INVALID_DEVICE_PARTITION_COUNT:
        return format(err, "CL_INVALID_DEVICE_PARTITION_COUNT");
      default:
        return format(err, "UNKNOWN OPENCL ERROR");
    }
  }

  void
  print_no_gpus_errmsg() {
    std::cerr << "error: no OpenCL-enabled GPUs have been found!" << std::endl;
  }

  unsigned int
  max_wgsize(GPUElement& gpu
           , unsigned int needed_kernel_resources) {
    cl_ulong max_local_mem;
    std::size_t gpu_max_wgsize;
    std::size_t psize;
    // get max. local memory
    check_error(clGetDeviceInfo(gpu.id
                              , CL_DEVICE_LOCAL_MEM_SIZE
                              , 0
                              , NULL
                              , &psize)
              , "clGetDeviceInfo");
    check_error(clGetDeviceInfo(gpu.id
                              , CL_DEVICE_LOCAL_MEM_SIZE
                              , psize
                              , &max_local_mem
                              , NULL)
              , "clGetDeviceInfo");
    // get max. work group size (1D)
    check_error(clGetDeviceInfo(gpu.id
                              , CL_DEVICE_MAX_WORK_GROUP_SIZE
                              , 0
                              , NULL
                              , &psize)
              , "clGetDeviceInfo");
    check_error(clGetDeviceInfo(gpu.id
                              , CL_DEVICE_MAX_WORK_GROUP_SIZE
                              , psize
                              , &gpu_max_wgsize
                              , NULL)
              , "clGetDeviceInfo");
    unsigned int max_wgsize = (unsigned int)
                                max_local_mem / needed_kernel_resources;
    return std::min(max_wgsize, (unsigned int) gpu_max_wgsize);
  }

  void
  pfn_notify(const char *errinfo
           , const void *private_info
           , size_t cb
           , void *user_data) {
    UNUSED(private_info);
    UNUSED(cb);
    UNUSED(user_data);
	  std::cerr << "OpenCL Error (via pfn_notify): "
              << errinfo
              << std::endl;
  }

  void
  check_error(cl_int err_code, const char* err_name) {
    if (err_code != CL_SUCCESS) {
      std::cerr << "error:  "
                << err_name
                << ": "
                << Tools::OCL::err_to_string(err_code) << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void
  cleanup_gpu(GPUElement& gpu) {
    check_error(clFinish(gpu.q), "cleanup: clFinish");
    for (auto& kv: gpu.buffers) {
      check_error(clReleaseMemObject(kv.second), "clReleaseMemObject");
    }
    for (auto& kv: gpu.kernels) {
      check_error(clReleaseKernel(kv.second), "clReleaseKernel");
    }
    check_error(clReleaseProgram(gpu.prog), "clReleaseProgram");
    check_error(clReleaseCommandQueue(gpu.q), "clReleaseCommandQueue");
    check_error(clReleaseContext(gpu.ctx), "clReleaseContext");
  }

} // end namespace Tools::OCL
} // end namespace Tools

