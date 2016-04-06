
#pragma once

#include <boost/program_options.hpp>

namespace Transs {

  /**
   * Check command line options, read input,
   * run transfer entropy computation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);

  /**
   * Steer OpenCL transfer entropy computation and
   * return results.
   */
  std::vector<std::vector<float>> 
  compute_transfer_entropies(const float* coords
                           , std::size_t n_rows
                           , std::size_t n_cols
                           , unsigned int tau
                           , std::vector<float> bandwidths
                           , unsigned int wgsize);

} // end namespace 'Transs'

