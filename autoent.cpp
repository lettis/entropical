
#include <cmath>

#include "autoent.hpp"
#include "tools.hpp"

#ifdef USE_CUDA
  #include "densities_cuda.hpp"
#else
  #include "densities_omp.hpp"
#endif


namespace AutoEnt {

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
    std::vector<double> auto_ent(n_cols, 0.0);
    for (unsigned int i=0; i < n_cols; ++i) {
      float h = bandwidths[i];
      std::vector<double> p_itau = combined_densities(coords
                                                    , n_rows
                                                    , {i}
                                                    , {h}
                                                    , {tau});
      std::vector<double> p_i_itau = combined_densities(coords
                                                      , n_rows
                                                      , {i, i}
                                                      , {h, h}
                                                      , {0, tau});
      for (unsigned int k=0; k < n_rows-tau; ++k) {
        if (p_i_itau[k] > 0) {
          auto_ent[i] += p_i_itau[k] * h * h
                           * logfunc(p_itau[k] * Tools::inv(p_i_itau[k] * h));
        }
      }
    }
    // output
    for (unsigned int i=0; i < n_cols; ++i) {
      Tools::IO::out() << " " << auto_ent[i] << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end AutoEnt::

