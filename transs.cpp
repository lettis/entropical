
#include "transs.hpp"
#include "tools.hpp"

#include <omp.h>

namespace Transs {

  void
  main(boost::program_options::variables_map args) {
    UNUSED(args);
  }

  std::vector<std::vector<float>> 
  compute_transfer_entropies(const float* coords
                           , std::size_t n_rows
                           , std::size_t n_cols
                           , unsigned int tau
                           , std::vector<float> bandwidths
                           , unsigned int wgsize) {
    UNUSED(coords);
    UNUSED(n_rows);
    UNUSED(n_cols);
    UNUSED(tau);
    UNUSED(bandwidths);
    UNUSED(wgsize);
    return {};
  }

} // end namespace 'Transs'

