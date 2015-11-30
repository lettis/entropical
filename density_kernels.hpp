#pragma once

#include <vector>
#include <tuple>

namespace Transs {
  namespace Epanechnikov {
    /**
     * compute time-lagged probabilities
     *   p(x_n), p(x_n, x_n+tau)
     * and additionally return list of real neighbors of x
     */
    std::tuple<float, float>
    time_lagged_probabilities(std::size_t n
                            , std::size_t tau
                            , const float* coords
                            , std::size_t n_rows
                            , std::size_t x
                            , float bandwidth_x
                            , const std::vector<std::size_t>& x_neighbor_boxes);

    /**
     * compute joint probabilities
     *  p(x_n, y_n),  p(x_n, y_n, x_n_tau),  p(x_n, y_n, y_n_tau)
     */
    std::tuple<float, float, float>
    joint_probabilities(std::size_t n
                      , std::size_t tau
                      , const float* coords
                      , std::size_t n_rows
                      , std::size_t x
                      , std::size_t y
                      , float bandwidth_x
                      , float bandwidth_y
                      , const std::vector<std::size_t> neighbors);
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

