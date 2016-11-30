
#include "hestimate.hpp"
#include "tools.hpp"

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
  

  namespace Amise {

    float
    Kernel::_amise(float h) {
      //TODO
    }

    float
    Kernel::_newton_optimized_h(float h0
                              , float delta_h
                              , unsigned int iter_max) {
      //TODO
    }

    float
    Kernel::operator()(float* coords
                     , std::size_t n_rows
                     , std::size_t i_col) {
      // initialize private members for minimization subroutines
      _coords = coords;
      _n_rows = n_rows;
      _i_col = i_col;
      //TODO: get first estimate for bandwidth from rule of thumb
      //TODO: minimize amise with newton's help
      //TODO: return optimal h
    }
  
    void
    main(boost::program_options::variables_map args) {
      Hestimate::main<Hestimate::Amise::Kernel>(args);
    }
  } // end Hestimate::Amise

} // end Hestimate

