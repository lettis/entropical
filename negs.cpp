
#include <cmath>

#include "negs.hpp"
#include "dens.hpp"
#include "tools.hpp"

namespace Negs {

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
    // compute densities on GPUs
    std::vector<std::vector<float>> densities;
    densities = Dens::compute_densities(selected_cols
                                      , coords
                                      , n_rows
                                      , bandwidths);
    // compute negentropies
    std::size_t n_sel_cols = selected_cols.size();
    std::size_t i, j;
    #pragma omp parallel for default(none)\
                             private(i,j)\
                             firstprivate(n_sel_cols,n_rows)\
                             shared(densities)\
                             collapse(2)
    for (j=0; j < n_sel_cols; ++j) {
      for (i=0; i < n_rows; ++i) {
        if (densities[j][i] > 0) {
          densities[j][i] *= std::log2(densities[j][i]);
        }
      }
    }
    // output: negentropies
    for (std::size_t j=0; j < n_sel_cols; ++j) {
      Tools::IO::out() << -1.0 * Tools::kahan_sum(densities[j]) << std::endl;
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Negs'

