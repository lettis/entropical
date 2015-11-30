
#include "tools.hpp"
#include "density_kernels.hpp"

namespace Transs {
  namespace Epanechnikov {
    std::tuple<float, float, std::vector<std::size_t>>
    time_lagged_probabilities(std::size_t n
                            , std::size_t tau
                            , const float* coords
                            , std::size_t n_rows
                            , std::size_t x
                            , float bandwidth_x
                            , const std::vector<std::size_t> x_neighbor_boxes) {
      float x_n = coords[x*n_rows+n];
      float x_n_tau = coords[x*n_rows+n+tau];
      float h_x_2 = POW2(bandwidth_x);
      // results
      float p_x = 0.0f;
      float p_xtau = 0.0f;
      std::vector<std::size_t> neighbors;
      // kernel
      auto epanechnikov = [](float u2) -> float {return 0.75 * (1-u2);};
      for (std::size_t i_neighbor: x_neighbor_boxes) {
        float u_x = POW2(x_n - coords[x*n_rows+i_neighbor]) / h_x_2;
        if (u_x <= 1) {
          float u_xtau = POW2(x_n_tau - coords[x*n_rows+i_neighbor]) / h_x_2;
          p_x += epanechnikov(u_x);
          p_xtau += epanechnikov(u_x) * epanechnikov(u_xtau);
          neighbors.push_back(i_neighbor);
        }
      }
      return std::make_tuple(p_x, p_xtau, neighbors);
    }



//TODO finish this code (new changes in main)
    std::tuple<float, float, float>
    joint_probabilities(std::size_t n
                      , std::size_t tau
                      , const float* coords
                      , std::size_t n_rows
                      , std::size_t y
                      , std::size_t x
                      , float bandwidth_y
                      , float bandwidth_x
                      , const std::vector<std::size_t> neighbors) {
      float x_n = coords[x*n_rows+n];
      float x_n_tau = coords[x*n_rows+n+tau];
      float y_n = coords[y*n_rows+n];
      float h_x_2 = POW2(bandwidth_x);
      float h_y_2 = POW2(bandwidth_y);
      auto epanechnikov = [](float u2) -> float {return 0.75 * (1-u2);};
      std::array<float, 4> P = {0.0, 0.0, 0.0, 0.0};
      for (std::size_t i_neighbor: x_neighbor_boxes) {
        float u_x_n = POW2(x_n - coords[x*n_rows+i_neighbor]) / h_x_2;
        if (u_x_n <= 1) {
          // P(x_n)
          P[0] += epanechnikov(u_x_n);
          float u_x_n_tau = POW2(x_n_tau - coords[x*n_rows+i_neighbor]) / h_x_2;
          float u_y_n = POW2(y_n - coords[y*n_rows+i_neighbor]) / h_y_2;
          if (u_x_n_tau <= 1) {
            // P(x_n, x_n_tau)
            P[1] += epanechnikov(u_x_n) * epanechnikov(u_x_n_tau);
          }
          if (u_y_n <= 1) {
            // P(x_n, y_n)
            P[2] += epanechnikov(u_x_n) * epanechnikov(u_y_n);
          }
          if (u_x_n_tau <= 1 && u_y_n <= 1) {
            // P(x_n, y_n, x_n_tau)
            P[3] += epanechnikov(u_x_n) * epanechnikov(u_y_n) * epanechnikov(u_x_n_tau);
          }
        }
      }
      return P;
    }
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

