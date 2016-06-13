
#include "densities_omp.hpp"
#include "tools.hpp"

#include <array>

#include <omp.h>

//#define NDEBUG
#include <assert.h>



namespace {


  float
  epanechnikov(float ref_val
             , float val
             , float h) {
    float u_squared = (ref_val-val) / h;
    u_squared *= u_squared;
    return (u_squared <= 1.0f ? 0.75 * (1-u_squared) : 0.0f) / h;
  };

  typedef std::vector<std::array<std::size_t, 2>> Boxes;

  Boxes
  boxes(const std::vector<float>& sorted_coords
      , std::size_t n_rows
      , float h) {
    Boxes bxs(n_rows);
    std::size_t i,k;
    #pragma omp parallel for default(none)\
                             private(i,k)\
                             firstprivate(n_rows,h)\
                             shared(bxs,sorted_coords)\
                             schedule(dynamic,8)
    for (i=0; i < n_rows; ++i) {
      // find min
      bxs[i][0] = i;
      for (k=i; k > 0; --k) {
        if (sorted_coords[i] - sorted_coords[k-1] > h) {
          bxs[i][0] = k-1;
          break;
        }
      }
      // find max
      bxs[i][1] = i;
      for (k=i+1; k < n_rows; ++k) {
        if (sorted_coords[k] - sorted_coords[i] > h) {
          bxs[i][1] = k;
          break;
        }
      }
    }
    return bxs;
  }

  std::vector<float>
  densities_1d(const std::vector<float>& sorted_coords
             , std::vector<float> h) {
    std::size_t n_rows = sorted_coords.size();
    std::vector<float> P(n_rows);
    std::size_t i,k;
    float ref_val, epa;
    bool in_range;
//    #pragma omp parallel for default(none)\
//                             private(ref_val,i,k,in_range,epa)\
//                             firstprivate(n_rows,h)\
//                             shared(sorted_coords,P)\
//                             schedule(dynamic,1)
    for (i=0; i < n_rows; ++i) {
      ref_val = sorted_coords[i];
      in_range = false;
      for (k=0; k < n_rows; ++k) {
        epa = epanechnikov(ref_val, sorted_coords[k], h[0]);
//        if (epa > 0) {
//          std::cerr << i << " " << epa << std::endl;
//        }
//std::cerr << epa << std::endl;
//        if (epa > 0.0f && (! in_range)) {
//          in_range = true;
//          P[i] = epa;
//        } else if (epa > 0.0f && (in_range)) {
          P[i] += epa;
//        } else if (in_range) {
//          break;
//        }
      }
//      P[i] /= n_rows;
    }
    return P;
  }
  
  std::vector<float>
  densities_2d(const std::vector<float>& sorted_coords
             , std::vector<float> h) {
    std::size_t n_rows = sorted_coords.size() / 2;
    std::vector<float> P(n_rows);
    std::size_t i,k;
    float ref_1, ref_2;
    Boxes bxs = boxes(sorted_coords, n_rows, h[0]);
    #pragma omp parallel for default(none)\
                             private(ref_1,ref_2,i,k)\
                             firstprivate(n_rows,h)\
                             shared(bxs,sorted_coords,P)
    for (i=0; i < n_rows; ++i) {
      ref_1 = sorted_coords[i];
      ref_2 = sorted_coords[n_rows+i];
      for (k=bxs[i][0]; k <= bxs[i][1]; ++k) {
        P[i] += epanechnikov(ref_1, sorted_coords[k], h[0])
              * epanechnikov(ref_2, sorted_coords[n_rows+k], h[1]);
      }
      P[i] /= n_rows;
    }
    return P;
  }
  
  std::vector<float>
  densities_3d(const std::vector<float>& sorted_coords
             , std::vector<float> h) {
    std::size_t n_rows = sorted_coords.size() / 3;
    std::vector<float> P(n_rows);
    std::size_t i,k;
    float ref_1, ref_2, ref_3;
    Boxes bxs = boxes(sorted_coords, n_rows, h[0]);
    #pragma omp parallel for default(none)\
                             private(ref_1,ref_2,ref_3,i,k)\
                             firstprivate(n_rows,h)\
                             shared(bxs,sorted_coords,P)
    for (i=0; i < n_rows; ++i) {
      ref_1 = sorted_coords[i];
      ref_2 = sorted_coords[n_rows+i];
      ref_3 = sorted_coords[2*n_rows+i];
      for (k=bxs[i][0]; k <= bxs[i][1]; ++k) {
        P[i] += epanechnikov(ref_1, sorted_coords[k], h[0])
              * epanechnikov(ref_2, sorted_coords[n_rows+k], h[1])
              * epanechnikov(ref_3, sorted_coords[2*n_rows+k], h[2]);
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
  std::size_t n_dim = i_col.size();
  assert(1 <= n_dim
      && n_dim <= 3
      && n_dim == h.size());
  std::vector<float> sorted_coords = Tools::dim1_sorted_coords(coords
                                                             , n_rows
                                                             , i_col);
  switch(n_dim) {
    case 1:
      return densities_1d(sorted_coords, h);
    case 2:
      return densities_2d(sorted_coords, h);
    case 3:
      return densities_3d(sorted_coords, h);
    default:
      // this should never happen!
      exit(EXIT_FAILURE);
  }
}
