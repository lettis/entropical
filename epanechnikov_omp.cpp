
#include "tools.hpp"
#include "epanechnikov_omp.hpp"

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
        std::array<float, N_PROBS>
        joint_probabilities_unnormalized(std::size_t n
                                       , std::size_t tau
                                       , const float* coords
                                       , std::size_t n_rows
                                       , std::size_t x
                                       , std::size_t y
                                       , const std::vector<float>& bandwidths
                                       , const std::vector<Transs::BoxedSearch::Boxes>& searchboxes) {
          std::array<float, N_PROBS> P;
          std::size_t ntau = n+tau;
          float x_n = coords[x*n_rows+n];
          float x_ntau = coords[x*n_rows+ntau];
          float y_n = coords[y*n_rows+n];
          float y_ntau = coords[y*n_rows+ntau];
          float hx_squared = POW2(bandwidths[x]);
          float hy_squared = POW2(bandwidths[x]);
          for (std::size_t i=0; i < N_PROBS; ++i) {
            P[i] = 0.0f;
          }
          auto epanechnikov_1d = [](float u_squared) -> float {return (u_squared <= 1.0f ? 0.75 * (1-u_squared) : 0.0f);};
          auto ux_squared = [&](float x1, float x2) -> float {return POW2(x1-x2) / hx_squared;};
          auto uy_squared = [&](float y1, float y2) -> float {return POW2(y1-y2) / hy_squared;};
          std::size_t ix=0, ixtau=0, iy=0, iytau=0;
          std::vector<std::size_t> neighbors_x = searchboxes[x].neighbors_of_state_ordered(n);
          std::vector<std::size_t> neighbors_y = searchboxes[y].neighbors_of_state_ordered(n);
          std::vector<std::size_t> neighbors_xtau = searchboxes[x].neighbors_of_state_ordered(ntau);
          std::vector<std::size_t> neighbors_ytau = searchboxes[y].neighbors_of_state_ordered(ntau);
          // the following loop works, because neighbors
          // are sorted in ascending order by default
          // as given by neighbors_of_state_ordered(...)
          while (ix < neighbors_x.size() && iy < neighbors_y.size()) {
            // time-lagged neighbors only interesting if same neighbors as without time-lag
            while (neighbors_xtau[ixtau] < neighbors_x[ix] && ixtau < neighbors_xtau.size()) {
              ++ixtau;
            }
            while (neighbors_ytau[iytau] < neighbors_y[iy] && iytau < neighbors_ytau.size()) {
              ++iytau;
            }
            bool propagate_x = false;
            bool propagate_y = false;
            std::array<float, N_PROBS> P_tmp = {0,0,0,0,0,0,0};
            // probabilities for x and time-lagged x
            if (neighbors_x[ix] <= neighbors_y[iy]) {
              propagate_x = true;
              P_tmp[X] = epanechnikov_1d(ux_squared(x_n, coords[x*n_rows+neighbors_x[ix]]));
              if (neighbors_xtau[ixtau] == neighbors_x[ix]) {
                P_tmp[X_XTAU] = P_tmp[X] * epanechnikov_1d(ux_squared(x_ntau, coords[x*n_rows+neighbors_xtau[ixtau]]));
              }
            }
            // probabilities for y and time-lagged y
            if (neighbors_y[iy] <= neighbors_x[ix]) {
              propagate_y = true;
              P_tmp[Y] = epanechnikov_1d(uy_squared(y_n, coords[y*n_rows+neighbors_y[iy]]));
              if (neighbors_ytau[iytau] == neighbors_y[iy]) {
                P_tmp[Y_YTAU] = P_tmp[Y] * epanechnikov_1d(uy_squared(y_ntau, coords[y*n_rows+neighbors_ytau[iytau]]));
              }
            }
            // joint probabilities
            if (propagate_x && propagate_y) {
              P_tmp[X_XTAU_Y] = P_tmp[X_XTAU] * P_tmp[Y];
              P_tmp[Y_YTAU_X] = P_tmp[Y_YTAU] * P_tmp[X];
              P_tmp[X_Y] = P_tmp[X] * P_tmp[Y];
            }
            // accumulate probs
            for (std::size_t i=0; i < N_PROBS; ++i) {
              P[i] += P_tmp[i];
            }
            // next iteration ...
            if (propagate_x) {
              ++ix;
            }
            if (propagate_y) {
              ++iy;
            }
          }
          return P;
        }
      } // end local namespace

      std::array<float, N_T>
      transfer_entropies(std::size_t tau
                       , const float* coords
                       , std::size_t n_rows
                       , std::size_t x
                       , std::size_t y
                       , const std::vector<float>& bandwidths
                       , const std::vector<Transs::BoxedSearch::Boxes>& searchboxes) {
        std::size_t i, n;
        // use double precision for robust large-number summation
        std::vector<double> T_xy(n_rows-tau, 0.0);
        std::vector<double> T_yx(n_rows-tau, 0.0);
        std::vector<std::vector<double>> P(N_PROBS, std::vector<double>(n_rows-tau));
        std::array<float, N_PROBS> P_tmp;
        #pragma omp parallel for default(none)\
                                 private(n,P_tmp,i)\
                                 firstprivate(n_rows,tau,x,y)\
                                 shared(coords,bandwidths,searchboxes,T_xy,T_yx,P)\
                                 schedule(dynamic,1)
        for (n=0; n < n_rows-tau; ++n) {
          P_tmp = joint_probabilities_unnormalized(n
                                                 , tau
                                                 , coords
                                                 , n_rows
                                                 , x
                                                 , y
                                                 , bandwidths
                                                 , searchboxes);
          if (P_tmp[X_XTAU_Y] > 0.0) {
            T_yx[n] = static_cast<double>(P_tmp[X_XTAU_Y] * log(P_tmp[X_XTAU_Y] * P_tmp[X] / P_tmp[X_Y] / P_tmp[X_XTAU]));
          }
          if (P_tmp[Y_YTAU_X] > 0.0) {
            T_xy[n] = static_cast<double>(P_tmp[Y_YTAU_X] * log(P_tmp[Y_YTAU_X] * P_tmp[Y] / P_tmp[X_Y] / P_tmp[Y_YTAU]));
          }
          // store probabilities for later renormalization
          for (i=0; i < N_PROBS; ++i) {
            P[i][n] = P_tmp[i];
          }
        }
        // sums of probabilities for renormalization of P (i.e. integral over P_i = 1)
        std::array<double, N_PROBS> sum_P;
        for (i=0; i < N_PROBS; ++i) {
          sum_P[i] = Tools::kahan_sum(P[i]);
        }
        std::array<float, N_T> _T;
        double tmp;
        // T_XY
        tmp = Tools::kahan_sum(T_xy) / n_rows / POW2(bandwidths[y]) / bandwidths[x];
        tmp += (n_rows-tau) * (log(sum_P[X_Y]) + log(sum_P[Y_YTAU]) - log(sum_P[Y_YTAU_X]) - log(sum_P[Y]));
        tmp /= sum_P[Y_YTAU_X];
        _T[XY] = static_cast<float>(tmp);
        // T_YX
        tmp = Tools::kahan_sum(T_yx) / n_rows / POW2(bandwidths[x]) / bandwidths[y];
        tmp += (n_rows-tau) * (log(sum_P[X_Y]) + log(sum_P[X_XTAU]) - log(sum_P[X_XTAU_Y]) - log(sum_P[X]));
        tmp /= sum_P[Y_YTAU_X];
        _T[YX] = static_cast<float>(tmp);
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

