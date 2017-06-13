
#include "negs.hpp"
#include "tools.hpp"

#ifdef USE_CUDA
  #include "densities_cuda.hpp"
#else
  #include "densities_omp.hpp"
#endif


namespace Negs {

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
    std::vector<float> negs(n_cols);
    for (unsigned int i=0; i < n_cols; ++i) {
      std::vector<double> p = sum1_normalized(
                                combined_densities(coords
                                                 , n_rows
                                                 , {i}
                                                 , {bandwidths[i]}));
      for (unsigned int k=0; k < n_rows; ++k) {
        negs[i] += logfunc(p[k]) * p[k];
      }
      negs[i] *= -1.0f;
    }
    // output
    for (unsigned int i=0; i < n_cols; ++i) {
      Tools::IO::out() << " " << negs[i];
    }
    Tools::IO::out() << "\n";
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Negs'

