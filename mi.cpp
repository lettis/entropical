
#include "mi.hpp"
#include "tools.hpp"

#ifdef USE_OPENCL
  #include "tools_opencl.hpp"
  #include "densities_opencl.hpp"
#else
  #include "densities_omp.hpp"
#endif


namespace Mi {

  void
  main(boost::program_options::variables_map args) {
    // Mutual information: I(X,Y) = \sum_x \sum_y p(x,y) log(p(x,y) / p(x) / p(y))
    using Tools::IO::selected_coords_bandwidths;
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
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
    // fill 1D prob dens cache
//TODO
//    std::map<std::size_t
//           , std::vector<float>> 1d_cache;
//    for (std::size_t i=0; i < n_cols; ++i) {
//      1d_cache[i] = combined_densities(coords
//                                     , n_rows
//                                     , {i}
//                                     , {bandwidths[i]});
//    }
    // compute MI from combined prob densities
    for (std::size_t i=0; i < n_cols; ++i) {
      for (std::size_t j=i+1; j < n_cols; ++j) {
        std::vector<float> dens_ij = combined_densities(coords
                                                      , n_rows
                                                      , {i
                                                        ,j}
                                                      , {bandwidths[i]
                                                       , bandwidths[j]});
        //TODO compute MI
        //TODO debug out: 2d densities
        
        for (std::size_t k=0; k < n_rows; ++k) {
          Tools::IO::out() << coords[i*n_rows+k]
                           << " "
                           << coords[j*n_rows+k]
                           << " "
                           << dens_ij[k]
                           << "\n";
        }
        
      }
    }

    // output: MI
    //TODO
  }

} // end namespace 'Mi'

