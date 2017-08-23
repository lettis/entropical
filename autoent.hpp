#pragma once

#include <boost/program_options.hpp>

namespace AutoEnt {

  /**
   * Check command line options, read input,
   * run transfer entropy computation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);

} // end AutoEnt::

