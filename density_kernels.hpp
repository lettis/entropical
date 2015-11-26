#pragma once

namespace Transs {
  namespace Epanechnikov {
    /**
     * compute joint probabilities
     *   p(x_n), p(x_n, x_n+tau), p(x_n, y_n), p(x_n, x_n+tau, y_n)
     */
    std::array<float, 4>
    joint_probabilities(std::size_t n
                      , std::size_t tau
                      , const std::vector<float>& y
                      , const std::vector<float>& x
                      , float bandwidth_y
                      , float bandwidth_x
                      , const std::vector<std::size_t>& x_neighbor_boxes);
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

