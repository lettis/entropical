#pragma once

namespace Transs {
  namespace BoxedSearch {
    class Boxes {
     public:
       //TODO change vector to 'coords' field with given column
      Boxes(const std::vector<float>& values
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

