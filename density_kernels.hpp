#pragma once

namespace Transs {
  namespace Epanechnikov {
    /**
     * compute time-lagged probabilities
     *   p(x_n), p(x_n, x_n+tau)
     * and additionally return list of real neighbors of x
     */
    std::tuple<float, float, std::vector<std::size_t>>
    time_lagged_probabilities(std::size_t n
                            , std::size_t tau
                            , const float* coords
                            , std::size_t n_rows
                            , std::size_t x
                            , float bandwidth_x
                            , const std::vector<std::size_t>& x_neighbor_boxes);

//    /**
//     * compute joint probabilities
//     *   p(x_n, y_n), p(x_n, x_n+tau, y_n)
//     */
//    std::array<float, 2>
//    joint_probabilities(std::size_t n
//                      , std::size_t tau
//                      , const float* coords
//                      , std::size_t n_rows
//                      , std::size_t y
//                      , std::size_t x
//                      , float bandwidth_y
//                      , float bandwidth_x
//                      , const std::vector<std::size_t>& x_neighbor_boxes);
//


  } // end namespace Transs::Epanechnikov
} // end namespace Transs

