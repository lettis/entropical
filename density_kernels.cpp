
#include "tools.hpp"
#include "density_kernels.hpp"

namespace Transs {
  namespace Epanechnikov {
    std::array<float, 4>
    joint_probabilities(std::size_t n
                      , std::size_t tau
                      , const std::vector<float>& y
                      , const std::vector<float>& x
                      , float bandwidth_y
                      , float bandwidth_x
                      , const std::vector<std::size_t>& x_neighbor_boxes) {
      float x_n = x[n];
      float x_n_tau = x[n+tau];
      float y_n = y[n];
      float h_x_2 = POW2(bandwidth_x);
      float h_y_2 = POW2(bandwidth_y);
      auto epanechnikov = [](float u2) -> float {return 0.75 * (1-u2);};
      std::array<float, 4> P = {0.0, 0.0, 0.0, 0.0};
      for (std::size_t i_neighbor: x_neighbor_boxes) {
        u_x_n = POW2(x_n - x[i_neighbor]) / h_x_2;
        if (u_x_n <= 1) {
          // P(x_n)
          P[0] += epanechnikov(u_x_n);
          u_x_n_tau = POW2(x_n_tau - y[i_neighbor]) / h_x_2;
          u_y_n = POW2(y_n - y[i_neighbor]) / h_y_2;
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
      P[0] /= (x.size()*bandwidth_x);
      P[1] /= (x.size()*POW2(bandwidth_x));
      P[2] /= (x.size()*bandwidth_x*bandwidth_y);
      P[3] /= (x.size()*POW2(bandwidth_x)*bandwidth_y);
      return P;
    }
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

