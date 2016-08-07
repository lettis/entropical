
#include "kldiv.hpp"
#include "tools.hpp"

#ifdef USE_CUDA
  #include "densities_cuda.hpp"
#else
  #include "densities_omp.hpp"
#endif


namespace KLDiv {

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
    // compute Kullback-Leibler divergence
    using Tools::sum1_normalized;
    std::vector<float> kldiv(n_cols*n_cols, 0.0f);
    for (unsigned int i=0; i < n_cols; ++i) {
      std::vector<float> p_i = sum1_normalized(
                                 combined_densities(coords
                                                  , n_rows
                                                  , {i}
                                                  , {bandwidths[i]}));
      for (unsigned int j=i+1; j < n_cols; ++j) {
        std::vector<float> p_j = sum1_normalized(
                                   combined_densities(coords
                                                    , n_rows
                                                    , {j}
                                                    , {bandwidths[j]}));
        for (unsigned int k=0; k < n_rows; ++k) {
          kldiv[i*n_cols+j] += p_i[k] * logfunc(p_i[k] / p_j[k]);
          kldiv[j*n_cols+i] += p_j[k] * logfunc(p_j[k] / p_i[k]);
        }
      }
    }
    // output
    for (unsigned int i=0; i < n_cols; ++i) {
      for (unsigned int j=0; j < n_cols; ++j) {
        Tools::IO::out() << " " << kldiv[i*n_cols+j];
      }
      Tools::IO::out() << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Mi'

