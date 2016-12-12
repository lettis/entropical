#pragma once

#include <boost/program_options.hpp>

namespace Hestimate {

  namespace Thumb {
    struct Kernel {
      /**
       * Implementation of Silverman's rule of thumb
       * for bandwidth estimation.
       */
      float
      operator()(float* coords
               , std::size_t n_rows
               , std::size_t i_col);
    };

    void
    main(boost::program_options::variables_map args);

  } // end Hestimate::Thumb


  namespace AmiseMin {

    struct Kernel {
      /**
       * Bandwidth estimation by iterative AMISE minimization.
       */
      float
      operator()(float* coords
               , std::size_t n_rows
               , std::size_t i_col);
     private:
      std::vector<float> _sorted_coords;
      // amise estimator for given bandwidth h
      float
      _amise(float h);
      // interval optimizer [min(AMISE)] for bandwidth
      float
      _interval_optimized_h(float prec
                          , float h_lower_bound
                          , float h_upper_bound
                          , unsigned int iter_max);
    };

    void
    main(boost::program_options::variables_map args);

  } // end Hestimate::AmiseMin


  /**
   * Check command line options, read input,
   * run bandwidth estimation kernel and
   * write output to file (or stdout).
   */
  template <typename KRNL>
  void
  main(boost::program_options::variables_map args);
  
} // end Hestimate

#include "hestimate.hxx"

