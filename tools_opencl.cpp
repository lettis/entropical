
#include "tools_opencl.hpp"

namespace Tools {
namespace OCL {

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

} // end namespace Tools::OCL
} // end namespace Tools

