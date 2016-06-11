
#include <algorithm>
#include <string>
#include <omp.h>

#include "dens.hpp"
#include "densities.hpp"
#include "tools.hpp"

namespace Dens {

  std::vector<std::vector<float>>
  compute_densities(std::vector<std::size_t> selected_cols
                  , const float* coords
                  , std::size_t n_rows
                  , std::vector<float> bandwidths) {
    std::vector<std::vector<float>> densities(selected_cols.size());
    // setup OpenCL environment
    std::vector<Tools::OCL::GPUElement> gpus = Tools::OCL::gpus();
    std::size_t n_gpus = gpus.size();
    if (n_gpus == 0) {
      Tools::OCL::print_no_gpus_errmsg();
      exit(EXIT_FAILURE);
    }
    // TODO: workgroup size is hard-coded to optimum for GeForce GTX-960.
    //       this has to be changed to a relative value of max(wgsize).
    unsigned int wgsize1d = 128;
    // compute densities on available GPUs
    unsigned int n_wg = prepare_gpus(gpus
                                   , wgsize1d
                                   , n_rows
                                   , 1);
    unsigned int j, thread_id;
    unsigned int n_selected_cols = selected_cols.size();
    #pragma omp parallel for default(none)\
                             private(j,thread_id)\
                             firstprivate(n_selected_cols,n_rows,\
                                          n_wg,wgsize1d)\
                             shared(coords,selected_cols,gpus,\
                                    densities,bandwidths)\
                             num_threads(n_gpus)\
                             schedule(dynamic,1)
    for (j=0; j < n_selected_cols; ++j) {
      thread_id = omp_get_thread_num();
      densities[j] = combined_densities(&gpus[thread_id]
                                      , coords
                                      , n_rows
                                      , {j}
                                      , {bandwidths[j]}
                                      , n_wg
                                      , wgsize1d);
    }
    for (Tools::OCL::GPUElement& gpu: gpus) {
      Tools::OCL::cleanup_gpu(&gpu);
    }
    // normalize densities
    unsigned int i;
    float sum;
    #pragma omp parallel for default(none)\
                             private(j,i,sum)\
                             firstprivate(n_selected_cols,n_rows)\
                             shared(densities)
    for (j=0; j < n_selected_cols; ++j) {
      sum = Tools::kahan_sum(densities[j]);
      for (i=0; i < n_rows; ++i) {
        densities[j][i] /= sum;
      }
    }
    return densities;
  }


  void
  main(boost::program_options::variables_map args) {
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    // read data
    std::vector<std::size_t> selected_cols;
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::tie(selected_cols, coords, n_rows, n_cols)
      = Tools::IO::selected_coords<float>(fname_input
                                        , args["columns"].as<std::string>());
    std::vector<float> bandwidths;
    using Tools::String::split;
    for (std::string h: split(args["bandwidths"].as<std::string>()
                            , ' '
                            , true)) {
      bandwidths.push_back(std::stof(h));
    }
    if (bandwidths.size() != selected_cols.size()) {
      std::cerr << "error: number of given bandwidth values does not match"
                         " the number of selected columns!" << std::endl;
      exit(EXIT_FAILURE);
    }
    // run computation
    std::vector<std::vector<float>> densities;
    densities = compute_densities(selected_cols
                                , coords
                                , n_rows
                                , bandwidths);
    // output: densities
    for (std::size_t i=0; i < n_rows; ++i) {
      for (std::size_t j=0; j < selected_cols.size(); ++j) {
        Tools::IO::out() << " " << densities[j][i];
      }
      Tools::IO::out() << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Dens'

