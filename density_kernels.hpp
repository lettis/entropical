#pragma once

namespace Transs {
  namespace Epanechnikov {
    std::array<float, 4>
    joint_probabilities(std::size_t n
                      , const std::vector<float>& y
                      , const std::vector<float>& x
                      , const std::vector<std::size_t> box_of_state
                      , const std::vector<std::vector<std::size_t>> boxes);
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

