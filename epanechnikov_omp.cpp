
#include "tools.hpp"
#include "density_kernels.hpp"

namespace Transs {
  namespace Epanechnikov {
    namespace OMP {
      namespace {
        /**
         * compute (joint) probabilities for given dimensions x,y and specific frame n
         *
         * output:
         * [  p(x_n)
         *  , p(x_n, x_n+tau)
         *  , p(y_n),
         *  , p(y_n, y_n+tau)
         *  , p(x_n, y_n)
         *  , p(x_n, y_n, x_n_tau)
         *  , p(x_n, y_n, y_n_tau) ]
         *
         */
        std::array<float, 7>
        joint_probabilities_unnormalized(std::size_t n
                                       , std::size_t tau
                                       , const float* coords
                                       , std::size_t n_rows
                                       , std::size_t x
                                       , std::size_t y
                                       , const std::vector<float>& bandwidths
                                       , const std::vector<Transs::BoxedSearch::Boxes>& searchboxes) {
          std::size_t ntau = n+tau;
          float x_n = coords[x*n_rows+n];
          float x_ntau = coords[x*n_rows+ntau];
          float y_n = coords[y*n_rows+n];
          float y_ntau = coords[y*n_rows+ntau];
          float hx_squared = POW2(bandwidths[x]);
          float hy_squared = POW2(bandwidths[x]);
          std::array<float, 7> P{0, 0, 0, 0, 0, 0, 0};
          auto epanechnikov_1d = [](float u_squared) -> float {return 0.75 * (1-u_squared);};
          auto ux_squared = [&](float x1, float x2) -> float {return POW2(x1-x2) / hx_squared;};
          auto uy_squared = [&](float y1, float y2) -> float {return POW2(y1-y2) / hy_squared;};
          std::size_t ix=0, ixtau=0, iy=0, iytau=0;
          //TODO loop through neighbors

        }
      } // end local namespace

      std::array<float, 2>
      transfer_entropies(std::size_t tau
                       , const float* coords
                       , std::size_t n_rows
                       , std::size_t x
                       , std::size_t y
                       , const std::vector<float>& bandwidths
                       , const std::vector<Transs::BoxedSearch::Boxes>& searchboxes) {
        std::size_t n;
        std::array<float, 7> P;
        // use double precision for robust large-number summation
        std::vector<double> T_xy(n_rows-tau, 0.0);
        std::vector<double> T_yx(n_rows-tau, 0.0);
        #pragma omp parallel for default(none)\
                                 private(n,P)\
                                 firstprivate(n_rows,tau,x,y)\
                                 shared(coords,bandwidths,searchboxes,T_xy,T_yx)\
                                 schedule(dynamic,1)
        for (n=0; n < n_rows-tau; ++n) {
          P = joint_probabilities_unnormalized(n
                                             , tau
                                             , coords
                                             , n_rows
                                             , x
                                             , y
                                             , bandwidths
                                             , searchboxes);
          if (P[X_XTAU_Y] > 0.0) {
            T_yx[n] = static_cast<double>(P[X_XTAU_Y] * log(P[X_XTAU_Y] * P[X] / P[X_Y] / P[X_XTAU]));
          }
          if (P[Y_YTAU_X] > 0.0) {
            T_xy[n] = static_cast<double>(P[Y_YTAU_X] * log(P[Y_YTAU_X] * P[Y] / P[X_Y] / P[Y_YTAU]));
          }
        }
        std::array<float, 2> _T;
        _T[XY] = static_cast<float>(Tools::kahan_sum(T_xy) / n_rows / POW2(bandwidths[y]) / bandwidths[x]);
        _T[YX] = static_cast<float>(Tools::kahan_sum(T_yx) / n_rows / POW2(bandwidths[x]) / bandwidths[y]);
        return _T;
      }


/* old stuff
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
*/
    } // end namespace Transs::Epanechnikov::OMP
  } // end namespace Transs::Epanechnikov
} // end namespace Transs

