
#include "boxed_search.hpp"

#include <algorithm>
#include <cmath>

namespace Transs {
  namespace BoxedSearch {
    Boxes::Boxes(const std::vector<float>& values
               , float box_size)
      : _n_values(values.size())
      , _box_of_state(_n_values) {
      auto min_max_values = std::minmax(values);
      float min_val = min_max_values.first;
      float max_val = min_max_values.second;
      _n_boxes = (std::size_t) ceil((max_val - min_val) / box_size);
      _boxes.resize(_n_boxes);
      // assign states to boxes
      for (std::size_t i=0; i < _n_values; ++i) {
        std::size_t i_box = floor((values[i] - min_val) / box_size);
        _boxes[i_box].push_back(i);
        _box_of_state[i] = i_box;
      }
      // pre-compute extended neighbor boxes
      for (std::size_t i_box=0; i_box < _n_boxes; ++i_box) {
        std::vector<std::size_t> neighbors;
        std::size_t i_box = _box_of_state[i];
        if (i_box == 0) {
          if (_n_boxes == 1) {
            neighbors = _boxes[0];
          } else {
            neighbors.reserve(_boxes[0].size() + _boxes[1].size());
            neighbors.insert(neighbors.end(), _boxes[0].begin(), _boxes[0].end());
            neighbors.insert(neighbors.end(), _boxes[1].begin(), _boxes[1].end());
          }
        } else if (i_box == _n_boxes-1) {
          neighbors.reserve(_boxes[_n_boxes-1].size() + _boxes[_n_boxes-2].size());
          neighbors.insert(neighbors.end(), _boxes[_n_boxes-1].begin(), _boxes[_n_boxes-1].end());
          neighbors.insert(neighbors.end(), _boxes[_n_boxes-2].begin(), _boxes[_n_boxes-2].end());
        } else {
          neighbors.reserve(_boxes[i_box-1].size() + _boxes[i_box].size() + _boxes[i_box+1].size());
          neighbors.insert(neighbors.end(), _boxes[i_box-1].begin(), _boxes[i_box-1].end());
          neighbors.insert(neighbors.end(), _boxes[i_box].begin(), _boxes[i_box].end());
          neighbors.insert(neighbors.end(), _boxes[i_box+1].begin(), _boxes[i_box+1].end());
        }
        _neighbors.push_back(neighbors);
      }
    }
    std::vector<std::size_t>
    Boxes::neighbors_of_state(std::size_t i) {
      return _neighbors[_box_of_state[i]];
    }
  } // end namespace Transs::BoxedSearch
} // end namespace Transs

