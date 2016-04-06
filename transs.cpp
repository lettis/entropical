
#include "transs.hpp"
#include "transs_opencl.hpp"
#include "tools.hpp"

#include <omp.h>

namespace Transs {

  void
  main(boost::program_options::variables_map args) {
    //TODO
    
//    bool verbose = args["verbose"].as<bool>();
//    // set output stream to file or STDOUT, depending on args
//    Tools::IO::set_out(args["output"].as<std::string>()); 
//    std::string fname_input = args["input"].as<std::string>();
    
//    // read coordinates
//    float* coords;
//    std::size_t n_rows;
//    std::size_t n_cols;
//    if (pc_max == 0) {
//      verbose && std::cout << "reading full dataset" << std::endl;
//      // read full data set
//      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C');
//    } else {
//      // read only until given column
//      verbose && std::cout << "reading dataset up to PC " << pc_max << std::endl;
//      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C', Tools::range<std::size_t>(0, pc_max, 1));
//    }
    
    
//    std::vector<std::vector<float>> T = compute_transfer_entropies(coords
//                                                                 , n_rows
//                                                                 , n_cols
//                                                                 , tau
//                                                                 , {}
//                                                                 , wgsize);
//    // output
//    for (std::size_t y=0; y < n_cols; ++y) {
//      for (std::size_t x=0; x < n_cols; ++x) {
//        Tools::IO::out() << " " << T[y][x];
//      }
//      Tools::IO::out() << "\n";
//    }
//
//
//    // memory cleanup
//    Tools::IO::free_coords(coords);
  }

  std::vector<std::vector<float>> 
  compute_transfer_entropies(const float* coords
                           , std::size_t n_rows
                           , std::size_t n_cols
                           , unsigned int tau
                           , std::vector<float> bandwidths
                           , unsigned int wgsize) {

    //TODO
    bool verbose = false;

    using namespace Transs::OCL;
    std::vector<std::vector<float>> T(n_cols, std::vector<float>(n_cols, 0.0));
    std::size_t x, y, thread_id;

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
      exit( EXIT_FAILURE);
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

