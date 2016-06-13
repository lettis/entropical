
#include "densities_omp.hpp"

//#define NDEBUG
#include <assert.h>

#include <array>

namespace {

  constexpr float
  epanechnikov(const float ref
             , const float val
             , const float h) {
    float u_squared = (ref-val) / h;
    u_squared *= u_squared;
    return (u_squared <= 1.0f ? 0.75 * (1-u_squared) : 0.0f) / h;
  };

  std::vector<float>
  densities_1d(const std::vector<float> coords
             , h) {
    std::size_t n_rows = coords.size();
    std::vector<float> P(n_rows);
  
    std::size_t i,k;
    float ref;
    for (i=0; i < n_rows; ++i) {
      ref = coords[i];
      for (k=0; k < n_rows; ++k) {
        P[i] += epanechnikov(ref, coords[k], h);
      }
      P[i] /= n_rows;
    }
    return P;
  }
  
  std::vector<float>
  densities_2d(const std::vector<float> coords_1
             , const std::vector<float> coords_2
             , std::array<float, 2> h) {
    std::size_t n_rows = coords_1.size();
    assert(n_rows == coords_2.size());
  
    std::size_t i,k;
    float ref_1, ref_2;
  
    for (i=0; i < n_rows; ++i) {
      ref_1 = coords_1[i];
      ref_2 = coords_2[i];
      for (k=0; k < n_rows; ++k) {
        P[i] += epanechnikov(ref_1, coords_1[k], h[0])
              * epanechnikov(ref_2, coords_2[k], h[1]);
      }
      P[i] /= n_rows;
    }
    return P;
  }
  
  std::vector<float>
  densities_3d(const std::vector<float> coords_1
             , const std::vector<float> coords_2
             , const std::vector<float> coords_3
             , std::array<float, 3> h) {
    std::size_t n_rows = coords_1.size();
    assert(n_rows == coords_2.size()
        && n_rows == coords_3.size());
  
    std::size_t i,k;
    float ref_1, ref_2, ref_3;
  
    for (i=0; i < n_rows; ++i) {
      ref_1 = coords_1[i];
      ref_2 = coords_2[i];
      ref_3 = coords_2[i];
      for (k=0; k < n_rows; ++k) {
        P[i] += epanechnikov(ref_1, coords_1[k], h[0])
              * epanechnikov(ref_2, coords_2[k], h[1])
              * epanechnikov(ref_3, coords_3[k], h[2]);
      }
      P[i] /= n_rows;
    }
    return P;
  }

} // end local namespace

std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<std::size_t> i_col
                 , std::vector<float> h) {
  std::size_t n_dim = i_cols.size();
  assert(1 <= n_dim
      && n_dim <= 3
      && n_dim == h.size());

  //TODO set vector data directly to coords field addresses

}
