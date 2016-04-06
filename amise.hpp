
#pragma once

#include <boost/program_options.hpp>

namespace Amise {

  /**
   * Check command line options, read input,
   * run AMISE computation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);
  
} // end namespace 'Amise'

