
#include <cmath>

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
    auto logfunc = Tools::select_log(args["bits"].as<bool>());
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
    std::vector<std::vector<double>> p(n_cols);
    std::vector<std::vector<double>> p_tau(n_cols);
    std::vector<double> transs(n_cols*n_cols, 0.0);
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
        std::vector<double> p_ij, p_ij_itau, p_ij_jtau;
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
        for (unsigned int k=0; k < n_rows-tau; ++k) {
          double te;
          if (p_ij_itau[k] > 0) {
            te = p_ij_itau[k]
                   * logfunc(p_ij_itau[k]
                           * p[i][k]
                           * Tools::inv(p_ij[k])
                           * Tools::inv(p_tau[i][k]));
            if (std::isfinite(te) && (p_ij_itau[k] <= 1.0)) {
              transs[i*n_cols+j] += te;
            } else {
              std::cerr << "strange values in TE("
                        << i+1 << ", " << j+1 << "): "
                        << "p_ij_itau = " << p_ij_itau[k] << ",  "
                        << "p_i = " << p[i][k] << ",  "
                        << "p_ij = " << p_ij[k] << ",  "
                        << "p_itau = " << p_tau[i][k] << ",  "
                        << "k = " << k << std::endl;
            }
          }
          if (p_ij_jtau[k] > 0) {
            te = p_ij_jtau[k]
                   * logfunc(p_ij_jtau[k]
                           * p[j][k]
                           * Tools::inv(p_ij[k])
                           * Tools::inv(p_tau[j][k]));
            if (std::isfinite(te) && (p_ij_jtau[k] <= 1.0)) {
              transs[j*n_cols+i] += te;
            } else {
              std::cerr << "strange values in TE("
                        << i+1 << ", " << j+1 << "): "
                        << "p_ij_jtau = " << p_ij_jtau[k] << ",  "
                        << "p_j = " << p[j][k] << ",  "
                        << "p_ij = " << p_ij[k] << ",  "
                        << "p_jtau = " << p_tau[j][k] << ",  "
                        << "k = " << k << std::endl;
            }
          }
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

