
#pragma once

#include <boost/program_options.hpp>

namespace Hestimate {

  /**
   * Implementation of Silverman's rule of thumb
   * for bandwidth estimation.
   */
  float
  silverman(float* coords
          , std::size_t n_rows
          , std::size_t i_col);

  /**
   * Check command line options, read input,
   * run bandwidth estimation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);
  
} // end namespace 'Hestimate'

