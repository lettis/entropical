
#pragma once

#include <boost/program_options.hpp>

namespace KLDiv {

  /**
   * Check command line options, read input,
   * run Kullback-Leibler divergence computation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);
  
} // end namespace 'KLDiv'

