
#include "hestimate.hpp"
#include "tools.hpp"

//TODO: CUDA?
#include "densities_omp.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>

namespace Hestimate {
  namespace Thumb {
  
    float
    Kernel::operator()(float* coords
                     , std::size_t n_rows
                     , std::size_t i_col) {
      using namespace boost::accumulators;
      using VarAcc = accumulator_set<float, features<tag::variance(lazy)>>;
      VarAcc acc;
      for (std::size_t i=0; i < n_rows; ++i) {
        acc(coords[i_col*n_rows+i]);
      }
      return std::pow(n_rows, -1.0/7.0)*sqrt(variance(acc));
    }
  
    void
    main(boost::program_options::variables_map args) {
      Hestimate::main<Hestimate::Thumb::Kernel>(args);
    }
  } // end Hestimate::Thumb
  

  namespace AmiseMin {

    float
    Kernel::_amise(float h) {
      unsigned int n_rows = _sorted_coords.size();
      float amise_estimate = 0.04
                           * std::pow(h, 0.2)
                           * epa_convolution(_sorted_coords, h);
      amise_estimate = std::pow(amise_estimate, 0.2);
      amise_estimate = amise_estimate
                      * 1.25
                      * std::pow(4.5, 0.8)
                      / std::pow(float(n_rows), 0.8);
      return amise_estimate;
    }

    float
    Kernel::_newton_optimized_h(float h
                              , float delta_h
                              // hard-coded magic:
                              //   max. 20 iterations for min. performance hit,
                              //   negative bandwidth for initial recursion
                              , unsigned int iter_max = 20
                              , float h_prev = -1.0) {
      if ((iter_max == 0)
       || (h_prev > 0 && abs(h-h_prev) < delta_h)  ) {
        // either max. iterations or convergence reached
        return h;
      } else {
        float y = _amise(h);
        float y1 = _amise(h+delta_h);
        float a = (y-y1)/delta_h;
        h_prev = h;
        h += y/a;
        // end-recursive iteration
        return _newton_optimized_h(h
                                 , delta_h
                                 , iter_max-1
                                 , h_prev);
      }
    }

    float
    Kernel::operator()(float* coords
                     , std::size_t n_rows
                     , std::size_t i_col) {
      // pre-sort and filter coords, store as private member for minimization
      _sorted_coords = Tools::dim1_sorted_coords(coords
                                               , n_rows
                                               , {(unsigned int) i_col});
      // get first estimate for bandwidth from rule of thumb
      float h0 = Hestimate::Thumb::Kernel()(coords
                                          , n_rows
                                          , i_col);
      float delta_h = 0.001 * h0;
      // minimize amise with newton's help
      return _newton_optimized_h(h0, delta_h);
    }
  
    void
    main(boost::program_options::variables_map args) {
      Hestimate::main<Hestimate::AmiseMin::Kernel>(args);
    }
  } // end Hestimate::AmiseMin

} // end Hestimate

