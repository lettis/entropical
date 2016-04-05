
#include "transs.hpp"
#include "transs_opencl.hpp"

namespace Transs {

  std::vector<std::vector<float>> 
  compute_transfer_entropies(const float* coords
                           , std::size_t n_rows
                           , std::size_t n_cols
                           , unsigned int tau
                           , std::vector<float> bandwidths
                           , unsigned int wgsize) {
    using namespace Transs::OCL;
    std::vector<std::vector<float>> T(n_cols, std::vector<float>(n_cols, 0.0));
    std::size_t x, y, thread_id;

    //TODO use bandwidths from external
    bandwidths.resize(n_cols);
    {
      using namespace boost::accumulators;
      using VarAcc = accumulator_set<float, features<tag::variance(lazy)>>;
      for (std::size_t j=0; j < n_cols; ++j) {
        VarAcc acc;
        for (std::size_t i=0; i < n_rows; ++i) {
          acc(coords[j*n_rows+i]);
        }
        bandwidths[j] = std::pow(n_rows, -1.0/7.0)*sqrt(variance(acc));
      }
    }
    // OpenCL setup
    unsigned int n_workgroups =
      (unsigned int) std::ceil(n_rows / ((float) wgsize));
    verbose && std::cout << "(n_workgroups, n, n_extended): "
                         << "(" << n_workgroups
                         << ", " << n_rows
                         << ", " << n_workgroups*wgsize << ")" << std::endl;
    std::vector<GPUElement> available_gpus = gpus();
    std::size_t n_gpus = available_gpus.size();
    if (n_gpus == 0) {
      std::cerr << "error: no GPUs found for OpenCL "
                   "transfer entropy computation" << std::endl;
      return EXIT_FAILURE;
    } else {
      verbose && std::cout << "computing transfer entropies on "
                           << n_gpus
                           << " GPUs" << std::endl;
    }
    std::string kernel_src = load_kernel_source("transs.cl");
    for(GPUElement& gpu: available_gpus) {
      setup_gpu(gpu, kernel_src, wgsize, n_workgroups, n_rows);
    }
    #pragma omp parallel for default(none)\
                             private(x,y,thread_id)\
                             firstprivate(tau,n_rows,n_cols,n_workgroups,wgsize)\
                             shared(coords,bandwidths,available_gpus,T)\
                             num_threads(n_gpus)\
                             collapse(2)\
                             schedule(dynamic, 1)
    for (x=0; x < n_cols; ++x) {
      for (y=0; y < n_cols; ++y) {
        if (x < y) {
          thread_id = omp_get_thread_num();
          std::pair<float, float> _T = transfer_entropies(available_gpus[thread_id]
                                                        , x
                                                        , y
                                                        , coords
                                                        , n_rows
                                                        , tau
                                                        , bandwidths
                                                        , wgsize
                                                        , n_workgroups);
          T[x][y] = _T.first;
          T[y][x] = _T.second;
          // T[x][x] == 0  by construction
        }
      }
    }
    // OpenCL cleanup
    for (GPUElement& gpu: available_gpus) {
      cleanup_gpu(gpu);
    }
    return T;
  }

} // end namespace 'Transs'

