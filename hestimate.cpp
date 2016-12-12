
#include "hestimate.hpp"
#include "tools.hpp"

//TODO: CUDA?
#include "densities_omp.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cmath>

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
      float amise_estimate = 0.04             // (1/5)^2
                           * 1/std::pow(h, 5) // 1/h^5
                           * epa_convolution(_sorted_coords, h);
      amise_estimate = std::pow(amise_estimate, 0.2);
      amise_estimate = amise_estimate
                      * 1.25
                      * std::pow(0.6, 0.8)
                      / std::pow(float(n_rows), 0.8);
      return amise_estimate;
    }

    float
    Kernel::_interval_optimized_h(float prec
                                , float h_lower_bound
                                , float h_upper_bound
                                , unsigned int iter_max = 20) {
      float delta_h = h_upper_bound - h_lower_bound;
      float h_pivot = h_lower_bound + 0.5*delta_h;
      if ((iter_max == 0)
       || (delta_h <= prec)) {
        return h_pivot;
      } else {
        float h_pivot_l = h_pivot - 0.25*delta_h;
        float h_pivot_h = h_pivot + 0.25*delta_h;
        float a_pivot_l = _amise(h_pivot_l);
        float a_pivot_h = _amise(h_pivot_h);

std::cerr << h_pivot_l << " " << a_pivot_l << std::endl;
std::cerr << h_pivot_h << " " << a_pivot_h << std::endl;

        if (a_pivot_l < a_pivot_h) {
          return _interval_optimized_h(prec
                                     , h_lower_bound
                                     , h_pivot
                                     , iter_max-1);
        } else {
          return _interval_optimized_h(prec
                                     , h_pivot
                                     , h_upper_bound
                                     , iter_max-1);
        }
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
      // minimize amise with newton's help
      return _interval_optimized_h(0.01*h0
                                 , 0.1*h0
                                 , 2*h0);
    }
  
    void
    main(boost::program_options::variables_map args) {
      Hestimate::main<Hestimate::AmiseMin::Kernel>(args);
    }
  } // end Hestimate::AmiseMin

} // end Hestimate

