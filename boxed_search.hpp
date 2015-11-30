#pragma once

#include <vector>

namespace Transs {
  namespace BoxedSearch {
    class Boxes {
     public:
      Boxes();

      Boxes(const float* coords
          , std::size_t n_rows
          , std::size_t selected_col
          , float box_size);

      std::vector<std::size_t>
      neighbors_of_state(std::size_t i);

      const std::size_t n_boxes() const {
        return _n_boxes;
      }

      const std::size_t n_values() const {
        return _n_values;
      }
     protected:
      std::size_t _n_boxes;
      std::size_t _n_values;
      std::vector<std::size_t> _box_of_state;
      std::vector<std::vector<std::size_t>> _boxes;
      std::vector<std::vector<std::size_t>> _neighbors;
    };

    std::vector<std::size_t>
    joint_neighborhood(std::vector n1
                     , std::vector n2);
  } // end namespace Transs::BoxedSearch
} // end namespace Transs

