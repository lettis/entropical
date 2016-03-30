
#define UNUSED(expr) (void)(expr)

#include "transs_opencl.hpp"

#include <vector>
#include <fstream>
#include <streambuf>
#include <iostream>

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

  void
  setup_gpu(GPUElement& gpu
          , std::string kernel_src
          , unsigned int wgsize
          , unsigned int n_workgroups
          , unsigned int n_rows) {
    unsigned int n_extended = n_workgroups * wgsize;
    cl_int err;
    // prepare kernel source
    kernel_src = std::string("#define WGSIZE ")
                 + std::to_string(wgsize)
                 + std::string("\n")
                 + kernel_src;
    const char* src = kernel_src.c_str();
    // create context
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
                                       , &src
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
      exit(EXIT_FAILURE);
    }
    // create kernel objects
    auto create_kernel = [&](std::string kname) -> void {
      gpu.kernels[kname] = clCreateKernel(gpu.prog
                                        , kname.c_str()
                                        , &err);
      check_error(err, "clCreateKernel");
    };
    create_kernel("partial_probs");
    create_kernel("collect_partials");
    create_kernel("compute_T");
    create_kernel("initialize_zero");
    // create buffers
    auto create_buffer = [&](std::string bname
                           , std::size_t bsize
                           , cl_mem_flags bflags) -> void {
      gpu.buffers[bname] = clCreateBuffer(gpu.ctx
                                        , bflags
                                        , bsize
                                        , NULL
                                        , &err);
      check_error(err, "clCreateBuffer");
    };
    // input coordinates
    create_buffer("i"
                , sizeof(float) * n_extended
                , CL_MEM_READ_ONLY);
    // input coordinates
    create_buffer("j"
                , sizeof(float) * n_extended
                , CL_MEM_READ_ONLY);
    // probabilities pre-reduced on workgroup level (single frame)
    create_buffer("Psingle"
                , sizeof(float) * 4*n_workgroups
                , CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
    // partially accumulated probabilities
    create_buffer("Pacc_partial"
                , sizeof(float) * 4*n_rows
                , CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
    // partially accumulated transfer entropy
    create_buffer("Tacc_partial"
                , sizeof(float) * n_rows
                , CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);
    // resulting transfer entropy
    create_buffer("T"
                , sizeof(float) * 2
                , CL_MEM_READ_WRITE);
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

  void
  pfn_notify(const char *errinfo
           , const void *private_info
           , size_t cb
           , void *user_data) {
    UNUSED(private_info);
    UNUSED(cb);
    UNUSED(user_data);
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
                   , std::size_t i
                   , std::size_t j
                   , const float* coords
                   , unsigned int n_rows
                   , unsigned int tau
                   , const std::vector<float> bandwidths
                   , unsigned int wgsize
                   , unsigned int n_workgroups) {
    unsigned int n_extended = n_workgroups * wgsize;
    std::size_t global_size = (std::size_t) n_extended;
    std::size_t local_size = (std::size_t) wgsize;
    // helpers for kernel arg settings & kernel invocation
    auto set_uint = [&] (std::string kname
                       , cl_int i
                       , unsigned int* ptr) -> void {
      check_error(clSetKernelArg(gpu.kernels[kname]
                               , i
                               , sizeof(unsigned int)
                               , ptr)
                , "clSetKernelArg: set_uint");
    };
    auto set_float = [&] (std::string kname
                        , cl_int i
                        , float* ptr) -> void {
      check_error(clSetKernelArg(gpu.kernels[kname]
                               , i
                               , sizeof(float)
                               , ptr)
                , "clSetKernelArg: set_float");
    };
    auto set_buf = [&] (std::string kname
                      , cl_int i
                      , std::string bname) -> void {
      check_error(clSetKernelArg(gpu.kernels[kname]
                               , i
                               , sizeof(gpu.buffers[bname])
                               , &gpu.buffers[bname])
                , "clSetKernelArg: set_buf");
    };
    auto nq_ndrange_kernel = [&] (std::string kname) -> void {
      check_error(clEnqueueNDRangeKernel(gpu.q
                                       , gpu.kernels[kname]
                                       , 1
                                       , NULL
                                       , &global_size
                                       , &local_size
                                       , 0
                                       , NULL
                                       , NULL)
                , "clEnqueueNDRangeKernel");
    };
    auto nq_task_kernel = [&] (std::string kname) -> void {
      check_error(clEnqueueTask(gpu.q
                              , gpu.kernels[kname]
                              , 0
                              , NULL
                              , NULL)
                , "clEnqueueTask");
    };
    auto nq_write = [&] (std::string bname, const float* ptr) -> void {
      check_error(clEnqueueWriteBuffer(gpu.q
                                     , gpu.buffers[bname]
                                     , CL_FALSE
                                     , 0
                                     , sizeof(float) * n_extended
                                     , ptr
                                     , 0
                                     , NULL
                                     , NULL)
                , "clEnqueueWriteBuffer");
    };
    // copy coords to buffers
    set_buf("initialize_zero", 0, "i");
    nq_ndrange_kernel("initialize_zero");
    nq_write("i", &coords[i*n_rows]);
    set_buf("initialize_zero", 0, "j");
    nq_ndrange_kernel("initialize_zero");
    nq_write("j", &coords[j*n_rows]);
    check_error(clFinish(gpu.q), "clFinish");

    // compute transfer entropy twice:
    //   i -> j (first run) and j -> i (second run)
    for (unsigned int idx: {0, 1}) {
      if (idx == 0) {
        set_buf("partial_probs", 0, "i");
        set_buf("partial_probs", 1, "j");
      } else {
        std::swap(i, j);
        set_buf("partial_probs", 1, "i");
        set_buf("partial_probs", 0, "j");
      }
      // (pre-)set kernel parameters and
      // run kernels for every frame (after tau)
      float h_inv_neg_i = -1.0f / bandwidths[i];
      float h_inv_neg_j = -1.0f / bandwidths[j];
      set_float("partial_probs", 5, &h_inv_neg_i);
      set_float("partial_probs", 6, &h_inv_neg_j);
      set_buf("partial_probs", 7, "Psingle");
      set_buf("collect_partials", 0, "Psingle");
      set_buf("collect_partials", 1, "Pacc_partial");
      set_buf("collect_partials", 2, "Tacc_partial");
      set_uint("collect_partials", 4, &n_rows);
      set_uint("collect_partials", 5, &n_workgroups);
      for (unsigned int k=tau; k < n_rows; ++k) {
        float ref_now_scaled = coords[j*n_rows+k] / bandwidths[j];
        float ref_prev_scaled = coords[j*n_rows+(k-1)] / bandwidths[j];
        float ref_tau_scaled = coords[i*n_rows+(k-tau)] / bandwidths[j];
        set_float("partial_probs", 2, &ref_now_scaled);
        set_float("partial_probs", 3, &ref_prev_scaled);
        set_float("partial_probs", 4, &ref_tau_scaled);
        nq_ndrange_kernel("partial_probs");
        unsigned int idx_partial = k - tau;
        set_uint("collect_partials", 3, &idx_partial);
        nq_task_kernel("collect_partials");
      }
      unsigned int n_partials = n_rows - tau;
      set_buf("compute_T", 0, "Pacc_partial");
      set_buf("compute_T", 1, "Tacc_partial");
      set_uint("compute_T", 2, &n_partials);
      set_uint("compute_T", 3, &n_workgroups);
      set_buf("compute_T", 4, "T");
      set_uint("compute_T", 5, &idx);
      nq_task_kernel("compute_T");
    }
    // collect results
    check_error(clFinish(gpu.q), "clFinish");
    float T[2];
    check_error(clEnqueueReadBuffer(gpu.q
                                  , gpu.buffers["T"]
                                  , CL_TRUE
                                  , 0
                                  , sizeof(float) * 2
                                  , T
                                  , 0
                                  , NULL
                                  , NULL)
              , "clEnqueueReadBuffer");
    check_error(clFinish(gpu.q), "clFinish");
    return {T[0], T[1]};
  }

} // end namespace Transs::OCL
} // end namespace Transs


