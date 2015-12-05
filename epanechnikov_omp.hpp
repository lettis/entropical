#pragma once

#include <vector>
#include <tuple>
#include <array>

#include "transs.hpp"

namespace Transs {
  namespace Epanechnikov {
    namespace OMP {
      /**
       * compute transfer entropies for given dimensions x,y.
       * output: [ T_xy, T_yx ]
       */
      std::array<float, 2>
      transfer_entropies(const float* coords
                       , std::size_t n_rows
                       , std::size_t x
                       , std::size_t y
                       , const std::vector<float>& bandwidths
                       , const std::vector<Transs::BoxedSearch::Boxes>& searchboxes);
    } // end namespace Transs::Epanechnikov::OMP
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

