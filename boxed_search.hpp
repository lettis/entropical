#pragma once

#include <vector>

namespace Transs {
  namespace BoxedSearch {
    class Boxes {
     public:
      Boxes(const float* coords
          , std::size_t n_rows
          , std::size_t selected_col
          , float box_size);
      std::vector<std::size_t>
      neighbors_of_state(std::size_t i);
     protected:
      std::size_t _n_boxes;
      std::size_t _n_values;
      std::vector<std::size_t> _box_of_state;
      std::vector<std::vector<std::size_t>> _boxes;
      std::vector<std::vector<std::size_t>> _neighbors;
    };
  } // end namespace Transs::BoxedSearch
} // end namespace Transs

