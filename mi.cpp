
#include "mi.hpp"
#include "tools.hpp"

#ifdef USE_CUDA
  #include "densities_cuda.hpp"
#else
  #include "densities_omp.hpp"
#endif


namespace Mi {

  void
  main(boost::program_options::variables_map args) {
    using Tools::IO::selected_coords_bandwidths;
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
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
    // compute mutual information
    using Tools::sum1_normalized;
    std::vector<float> mutinf(n_cols*n_cols);
    for (unsigned int i=0; i < n_cols; ++i) {
      std::vector<double> p_i = sum1_normalized(
                                  combined_densities(coords
                                                   , n_rows
                                                   , {i}
                                                   , {bandwidths[i]}));
      for (unsigned int j=i+1; j < n_cols; ++j) {
        std::vector<double> p_j = sum1_normalized(
                                    combined_densities(coords
                                                     , n_rows
                                                     , {j}
                                                     , {bandwidths[j]}));
        std::vector<double> p_ij = sum1_normalized(
                                     combined_densities(coords
                                                      , n_rows
                                                      , {i
                                                       , j}
                                                      , {bandwidths[i]
                                                       , bandwidths[j]}));
        mutinf[i*n_cols+j] = 0.0f;
        for (unsigned int k=0; k < n_rows; ++k) {
          mutinf[i*n_cols+j] += p_ij[k] * logfunc(p_ij[k] / p_i[k] / p_j[k]);
        }
        mutinf[j*n_cols+i] = mutinf[i*n_cols+j];
      }
    }
    // output
    for (unsigned int i=0; i < n_cols; ++i) {
      for (unsigned int j=0; j < n_cols; ++j) {
        Tools::IO::out() << " " << mutinf[i*n_cols+j];
      }
      Tools::IO::out() << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Mi'

