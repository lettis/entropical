
#include <algorithm>
#include <string>
#include <omp.h>
#include <vector>

#include "dens.hpp"

#ifdef USE_CUDA
  #include "densities_cuda.hpp"
#else
  #include "densities_omp.hpp"
#endif

#include "tools.hpp"

namespace Dens {

  std::vector<std::vector<float>>
  compute_densities_1d(std::vector<std::size_t> selected_cols
                     , const float* coords
                     , std::size_t n_rows
                     , std::vector<float> bandwidths) {
    std::vector<std::vector<float>> densities(selected_cols.size());
    unsigned int n_selected_cols = selected_cols.size();
    unsigned int j;
    //// compute probability densities with OpenMP on CPU
    for (unsigned int j=0; j < n_selected_cols; ++j) {
      densities[j] = combined_densities(coords
                                      , n_rows
                                      , {j}
                                      , {bandwidths[j]}
                                      , {0});
    }
    // normalize densities
    #pragma omp parallel for\
      default(none)\
      private(j)\
      firstprivate(n_selected_cols)\
      shared(densities)
    for (j=0; j < n_selected_cols; ++j) {
      densities[j] = Tools::sum1_normalized(densities[j]);
    }
    return densities;
  }

  std::tuple<std::vector<std::vector<float>>, std::vector<std::string>>
  compute_densities_nd(std::vector<std::size_t> selected_cols
                     , const float* coords
                     , std::size_t n_rows
                     , std::vector<float> bandwidths
                     , unsigned int dim_kernel
                     , std::vector<unsigned int> taus) {
    std::vector<std::string> labels;
    std::vector<std::vector<float>> densities;
    unsigned int n_selected_cols = selected_cols.size();
    // construct indices
    std::vector<std::vector<unsigned int>> indices;
    std::vector<std::vector<float>> hs;
    for (unsigned int i=0; i < n_selected_cols; ++i) {
      for (unsigned int j=i+1; j < n_selected_cols; ++j) {
        if (dim_kernel == 2) {
          indices.push_back({i
                            ,j});
          hs.push_back({bandwidths[i]
                      , bandwidths[j]});
        } else {
          for (unsigned int k=j+1; k < n_selected_cols; ++k) {
            indices.push_back({i
                              ,j
                              ,k});
            hs.push_back({bandwidths[i]
                        , bandwidths[j]
                        , bandwidths[k]});
          }
        }
      }
    }
    // compute probs
    for (unsigned int i=0; i < indices.size(); ++i) {
      std::vector<float> dens = combined_densities(coords
                                                 , n_rows
                                                 , indices[i]
                                                 , hs[i]
                                                 , taus);
      densities.push_back(dens);
      std::string lbl = std::to_string(selected_cols[indices[i][0]])
                      + "_"
                      + std::to_string(selected_cols[indices[i][1]]);
      if (dim_kernel == 3) {
        lbl += "_" + std::to_string(selected_cols[indices[i][2]]);
      }
      labels.push_back(lbl);
    }
    // normalize densities
    unsigned int j;
    #pragma omp parallel for\
      default(none)\
      private(j)\
      shared(densities)
    for (j=0; j < densities.size(); ++j) {
      densities[j] = Tools::sum1_normalized(densities[j]);
    }
    return std::make_tuple(densities, labels);
  }

  void
  main(boost::program_options::variables_map args) {
    using Tools::IO::selected_coords_bandwidths;
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    unsigned int dim_kernel = args["dims"].as<unsigned int>();
    if (dim_kernel == 0 || dim_kernel > 3) {
      std::cerr << "error: only allowed kernel dimensions are 1, 2 or 3."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    // read data
    std::vector<std::size_t> selected_cols;
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::vector<float> bandwidths;
    std::tie(selected_cols
           , coords
           , n_rows
           , n_cols
           , bandwidths)
      = selected_coords_bandwidths<float>(fname_input
                                        , args["columns"].as<std::string>()
                                        , args["bandwidths"].as<std::string>());
    // run computation
    std::vector<std::vector<float>> densities;
    if (dim_kernel == 1) {
      if (args["taus"].as<std::string>() != "") {
        std::cerr << "error: definition of lagtimes (taus) for 1d probability "
                  << "density does not make sense."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      densities = compute_densities_1d(selected_cols
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
    } else {
      std::vector<unsigned int> taus(dim_kernel, 0);
      std::string s_taus = args["taus"].as<std::string>();
      if (s_taus != "") {
        std::vector<std::string> v_s_taus = Tools::String::split(s_taus
                                                               , ' '
                                                               , true);
        if (v_s_taus.size() != dim_kernel) {
          std::cerr << "error: number of lagtime parameters does not match "
                    << "specified kernel dimension!"
                    << std::endl;
          exit(EXIT_FAILURE);
        }
        for (unsigned int i=0; i < dim_kernel; ++i) {
          taus[i] = std::stoul(v_s_taus[i]);
        }
      }
      std::vector<std::string> labels;
      std::tie(densities, labels) = compute_densities_nd(selected_cols
                                                       , coords
                                                       , n_rows
                                                       , bandwidths
                                                       , dim_kernel
                                                       , taus);
      for (std::size_t j=0; j < labels.size(); ++j) {
        std::string lbl = labels[j];
        for (std::size_t i=0; i < densities[j].size(); ++i) {
          Tools::IO::out() << lbl << " " << densities[j][i] << "\n";
        }
      }
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Dens'

