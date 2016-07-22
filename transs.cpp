
#include "transs.hpp"
#include "tools.hpp"

#ifdef USE_CUDA
  #include "densities_cuda.hpp"
#else
  #include "densities_omp.hpp"
#endif


namespace Transs {

  void
  main(boost::program_options::variables_map args) {
    using Tools::IO::selected_coords_bandwidths;
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    unsigned int tau = args["tau"].as<unsigned int>();
    //TODO log vs log2
    std::vector<std::size_t> selected_cols;
    // get coordinates
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
    // compute transfer entropies
    using Tools::sum1_normalized;
    std::vector<std::vector<float>> p(n_cols);
    std::vector<std::vector<float>> p_tau(n_cols);
    std::vector<float> transs(n_cols*n_cols, 0.0f);
    for (unsigned int i=0; i < n_cols; ++i) {
      p[i] = sum1_normalized(
              combined_densities(coords
                               , n_rows
                               , {i}
                               , {bandwidths[i]}));
      p_tau[i] = sum1_normalized(
                   combined_densities(coords
                                    , n_rows
                                    , {i
                                     , i}
                                    , {bandwidths[i]
                                     , bandwidths[i]}
                                    , {0
                                     , tau}));
    }
    for (unsigned int i=0; i < n_cols; ++i) {
      for (unsigned int j=i+1; j < n_cols; ++j) {
        std::vector<float> p_ij, p_ij_itau, p_ij_jtau;
        p_ij = sum1_normalized(
                 combined_densities(coords
                                  , n_rows
                                  , {i
                                   , j}
                                  , {bandwidths[i]
                                   , bandwidths[j]}));
        p_ij_itau = sum1_normalized(
                      combined_densities(coords
                                       , n_rows
                                       , {i
                                        , i
                                        , j}
                                       , {bandwidths[i]
                                        , bandwidths[i]
                                        , bandwidths[j]}
                                       , {0
                                        , tau
                                        , 0}));
        p_ij_jtau = sum1_normalized(
                      combined_densities(coords
                                       , n_rows
                                       , {i
                                        , j
                                        , j}
                                       , {bandwidths[i]
                                        , bandwidths[j]
                                        , bandwidths[j]}
                                       , {0
                                        , 0
                                        , tau}));
        for (unsigned int k=0; k < n_rows; ++k) {
          transs[i*n_cols+j] += p_ij_itau[k] * log(p_ij_itau[k] * p[i][k]
                                                 / (p_ij[k] * p_tau[i][k]));
          transs[j*n_cols+i] += p_ij_jtau[k] * log(p_ij_jtau[k] * p[j][k]
                                                 / (p_ij[k] * p_tau[j][k]));
        }
      }
    }
    // output
    for (unsigned int i=0; i < n_cols; ++i) {
      for (unsigned int j=0; j < n_cols; ++j) {
        Tools::IO::out() << " " << transs[i*n_cols+j];
      }
      Tools::IO::out() << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Transs'

