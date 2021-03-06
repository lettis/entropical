
#pragma once

#include <boost/program_options.hpp>

namespace Mi {

  /**
   * Check command line options, read input,
   * run mutual information computation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);
  
} // end namespace 'Mi'

