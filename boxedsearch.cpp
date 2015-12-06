
#include "boxedsearch.hpp"

#include <algorithm>
#include <iterator>
#include <cmath>

#include <iostream>

namespace Transs {
  namespace BoxedSearch {
    Boxes::Boxes() {
    }

    Boxes::Boxes(const float* coords
               , std::size_t n_rows
               , std::size_t selected_col
               , float box_size)
      : _n_values(n_rows)
      , _box_of_state(n_rows) {
      float min_val = std::numeric_limits<float>::max();
      float max_val = std::numeric_limits<float>::min();
      for (std::size_t i=0; i < n_rows; ++i) {
        min_val = std::min(min_val, coords[selected_col*n_rows+i]);
        max_val = std::max(max_val, coords[selected_col*n_rows+i]);
      }
      _n_boxes = (std::size_t) ceil((max_val - min_val) / box_size);
      _boxes.resize(_n_boxes);
      // assign states to boxes
      for (std::size_t i=0; i < _n_values; ++i) {
        std::size_t i_box = floor((coords[selected_col*n_rows+i] - min_val) / box_size);
        _boxes[i_box].push_back(i);
        _box_of_state[i] = i_box;
      }
      // pre-compute extended neighbor boxes
      for (std::size_t i_box=0; i_box < _n_boxes; ++i_box) {
        std::vector<std::size_t> neighbors;
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
      // sort extended neighbor boxes
      for (std::vector<std::size_t>& v: _neighbors) {
        std::sort(v.begin(), v.end());
      }
    }

    std::vector<std::size_t>
    Boxes::neighbors_of_state_ordered(std::size_t i) const {
      return _neighbors[_box_of_state[i]];
    }

    std::vector<std::size_t>
    joint_neighborhood(std::vector<std::size_t> n1
                     , std::vector<std::size_t> n2) {
      std::vector<std::size_t> joint_nh;
      std::set_intersection(n1.begin(), n1.end(), n2.begin(), n2.end(), std::back_inserter(joint_nh));
      return joint_nh;
    }
  } // end namespace Transs::BoxedSearch
} // end namespace Transs

