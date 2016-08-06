
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
  boxes(const float* coords
      , std::size_t i_col
      , const std::vector<float>& sorted_coords
      , std::size_t n_rows
      , float h) {
    Boxes bxs(n_rows);
    float ref_val;
    std::size_t i,k;
    #pragma omp parallel for default(none)\
                             private(ref_val,i,k)\
                             firstprivate(n_rows,h,i_col)\
                             shared(bxs,sorted_coords,coords)\
                             schedule(dynamic,8)
    for (i=0; i < n_rows; ++i) {
      ref_val = coords[i_col*n_rows+i];
      // find min
      for (k=0; k < n_rows; ++k) {
        if (ref_val - sorted_coords[k] < h) {
          bxs[i][0] = k;
          break;
        }
      }
      // find max
      for (k=n_rows; k > 0; --k) {
        if (sorted_coords[k-1] - ref_val < h) {
          bxs[i][1] = k-1;
          break;
        }
      }
    }
    return bxs;
  }

  std::vector<float>
  densities_1d(const float* coords
             , std::vector<unsigned int> i_col
             , const std::vector<float>& sorted_coords
             , std::vector<float> h) {
    std::size_t n_rows = sorted_coords.size();
    std::vector<float> P(n_rows, 0.0f);
    std::size_t i,k;
    float ref_val;
    Boxes bxs = boxes(coords
                    , i_col[0]
                    , sorted_coords
                    , n_rows
                    , h[0]);
    #pragma omp parallel for default(none)\
                             private(ref_val,i,k)\
                             firstprivate(n_rows,h,i_col)\
                             shared(coords,sorted_coords,P,bxs)\
                             schedule(dynamic,1)
    for (i=0; i < n_rows; ++i) {
      ref_val = coords[i_col[0]*n_rows+i];
      for (k=bxs[i][0]; k <= bxs[i][1]; ++k) {
        P[i] += epanechnikov(ref_val, sorted_coords[k], h[0]);
      }
      P[i] /= n_rows;
    }
    return P;
  }
  
  std::vector<float>
  densities_2d(const float* coords
             , std::vector<unsigned int> i_col
             , const std::vector<float>& sorted_coords
             , std::vector<float> h) {
    std::size_t n_rows = sorted_coords.size() / 2;
    std::vector<float> P(n_rows);
    Boxes bxs = boxes(coords
                    , i_col[0]
                    , sorted_coords
                    , n_rows
                    , h[0]);
    std::size_t i,k;
    float ref_1, ref_2;
    #pragma omp parallel for default(none)\
                             private(ref_1,ref_2,i,k)\
                             firstprivate(i_col,n_rows,h)\
                             shared(bxs,sorted_coords,P,coords)
    for (i=0; i < n_rows; ++i) {
      ref_1 = coords[i_col[0]*n_rows+i];
      ref_2 = coords[i_col[1]*n_rows+i];
      for (k=bxs[i][0]; k <= bxs[i][1]; ++k) {
        P[i] += epanechnikov(ref_1, sorted_coords[k], h[0])
              * epanechnikov(ref_2, sorted_coords[n_rows+k], h[1]);
      }
      P[i] /= n_rows;
    }
    return P;
  }
  
  std::vector<float>
  densities_3d(const float* coords
             , std::vector<unsigned int> i_col
             , const std::vector<float>& sorted_coords
             , std::vector<float> h) {
    std::size_t n_rows = sorted_coords.size() / 3;
    std::vector<float> P(n_rows);
    Boxes bxs = boxes(coords
                    , i_col[0]
                    , sorted_coords
                    , n_rows
                    , h[0]);
    std::size_t i,k;
    float ref_1, ref_2, ref_3;
    #pragma omp parallel for default(none)\
                             private(ref_1,ref_2,ref_3,i,k)\
                             firstprivate(i_col,n_rows,h)\
                             shared(bxs,sorted_coords,P,coords)
    for (i=0; i < n_rows; ++i) {
      ref_1 = coords[i_col[0]*n_rows+i];
      ref_2 = coords[i_col[1]*n_rows+i];
      ref_3 = coords[i_col[2]*n_rows+i];
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
                 , std::vector<unsigned int> i_cols
                 , std::vector<float> h) {
  unsigned int n_dim = i_cols.size();
  std::vector<unsigned int> tau;
  switch(n_dim) {
    case 1:
      tau = {0};
      break;
    case 2:
      tau = {0, 0};
      break;
    case 3:
      tau = {0, 0, 0};
      break;
    default:
      std::cerr << "error: unsupported number of dimensions. this should "
                << "never happen!"
                << std::endl;
      exit(EXIT_FAILURE);
  }
  return combined_densities(coords
                          , n_rows
                          , i_cols
                          , h
                          , tau);
}

std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<unsigned int> i_cols
                 , std::vector<float> h
                 , std::vector<unsigned int> tau) {
//TODO: include tau
  std::size_t n_dim = i_cols.size();
  assert(1 <= n_dim
      && n_dim <= 3
      && n_dim == h.size());
  std::vector<float> sorted_coords = Tools::dim1_sorted_coords(coords
                                                             , n_rows
                                                             , i_cols);
  switch(n_dim) {
    case 1:
      return densities_1d(coords
                        , i_cols
                        , sorted_coords
                        , h);
    case 2:
      return densities_2d(coords
                        , i_cols
                        , sorted_coords
                        , h);
    case 3:
      return densities_3d(coords
                        , i_cols
                        , sorted_coords
                        , h);
    default:
      // this should never happen!
      exit(EXIT_FAILURE);
  }
}
