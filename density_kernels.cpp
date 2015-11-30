
#include "tools.hpp"
#include "density_kernels.hpp"

namespace Transs {
  namespace Epanechnikov {
    std::tuple<float, float>
    time_lagged_probabilities(std::size_t n
                            , std::size_t tau
                            , const float* coords
                            , std::size_t n_rows
                            , std::size_t x
                            , float bandwidth_x
                            , const std::vector<std::size_t>& x_neighbor_boxes) {
      float x_n = coords[x*n_rows+n];
      float x_n_tau = coords[x*n_rows+n+tau];
      float h_x_2 = POW2(bandwidth_x);
      // results
      float p_x = 0.0f;
      float p_xtau = 0.0f;
      // kernel
      auto epanechnikov = [](float u2) -> float {return 0.75 * (1-u2);};
      for (std::size_t i_neighbor: x_neighbor_boxes) {
        float u_x = POW2(x_n - coords[x*n_rows+i_neighbor]) / h_x_2;
        if (u_x <= 1) {
          float u_xtau = POW2(x_n_tau - coords[x*n_rows+i_neighbor]) / h_x_2;
          p_x += epanechnikov(u_x);
          p_xtau += epanechnikov(u_x) * epanechnikov(u_xtau);
        }
      }
      return std::make_tuple(p_x, p_xtau);
    }

    std::tuple<float, float, float>
    joint_probabilities(std::size_t n
                      , std::size_t tau
                      , const float* coords
                      , std::size_t n_rows
                      , std::size_t x
                      , std::size_t y
                      , float bandwidth_x
                      , float bandwidth_y
                      , const std::vector<std::size_t> neighbors) {
      float x_n = coords[x*n_rows+n];
      float y_n = coords[y*n_rows+n];
      float x_n_tau = coords[x*n_rows+n+tau];
      float y_n_tau = coords[y*n_rows+n+tau];
      float h_x_2 = POW2(bandwidth_x);
      float h_y_2 = POW2(bandwidth_y);
      // results
      float p_xy = 0.0f;
      float p_x_xtau_y = 0.0f;
      float p_y_ytau_x = 0.0f;
      // kernel
      auto epanechnikov = [](float u2) -> float {return 0.75 * (1-u2);};
      float u_x_n;
      float u_y_n;
      float u_x_n_tau;
      float u_y_n_tau;
      for (std::size_t i_neighbor: neighbors) {
        u_x_n = POW2(x_n - coords[x*n_rows+i_neighbor]) / h_x_2;
        u_y_n = POW2(y_n - coords[y*n_rows+i_neighbor]) / h_y_2;
        if (u_x_n <= 1 && u_y_n <= 1) {
          u_x_n_tau = POW2(x_n_tau - coords[x*n_rows+i_neighbor]) / h_x_2;
          u_y_n_tau = POW2(y_n_tau - coords[y*n_rows+i_neighbor]) / h_y_2;
          if (u_x_n_tau <= 1) {
            p_x_xtau_y += epanechnikov(u_x_n) * epanechnikov(u_x_n_tau) * epanechnikov(u_y_n);
          }
          if (u_y_n_tau <= 1) {
            p_y_ytau_x += epanechnikov(u_y_n) * epanechnikov(u_y_n_tau) * epanechnikov(u_x_n);
          }
          p_xy += epanechnikov(u_x_n) * epanechnikov(u_x_n);
        }
      }
      return std::make_tuple(p_xy, p_x_xtau_y, p_y_ytau_x);
    }
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

