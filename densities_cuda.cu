
#include "densities_common.hpp"
#include "densities_cuda.hpp"

#include "density_1d.cuh"
#include "density_2d.cuh"
#include "density_3d.cuh"

#include "epa_conv.cuh"

#include <functional>
#include <algorithm>

#define BSIZE 128

std::vector<double>
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

std::vector<double>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<unsigned int> i_cols
                 , std::vector<float> h
                 , std::vector<unsigned int> tau) {
  // select coords according to tau values
  std::vector<float> sel_coords = prob_dens_coord_prep(coords
                                                     , n_rows
                                                     , i_cols
                                                     , h
                                                     , tau
                                                 // row-major result?
                                                     , true);
  unsigned int n_dim = i_cols.size();

  std::function<
    std::vector<float>(float*
                     , unsigned int
                     , std::vector<float>)> dens_func;
  switch(n_dim) {
  case 1:
    dens_func = &density_1d<float
                          , float
                          , BSIZE>;
    break;
  case 2:
    dens_func = &density_2d<float
                          , float
                          , BSIZE>;
    break;
  case 3:
    dens_func = &density_3d<float
                          , float
                          , BSIZE>;
    break;
  default:
    std::cerr << "error: in # of dim! this should never happen!" << std::endl;
    exit(EXIT_FAILURE);
  }
  // pre-invert bandwidths
  std::vector<float> h_inv(n_dim);
  for (unsigned int n=0; n < n_dim; ++n) {
    h_inv[n] = 1.0f/h[n];
  }
  // convert densities from GPU to double
  std::vector<float> flt_dens = dens_func(sel_coords.data()
                                        , sel_coords.size() / n_dim
                                        , h_inv);
  std::vector<double> dbl_dens(flt_dens.begin()
                             , flt_dens.end());
  // normalize
  std::transform(dbl_dens.begin()
               , dbl_dens.end()
               , dbl_dens.begin()
               , [n_rows] (double x) -> double {
                   return x / (double) n_rows;
                 });
  return dbl_dens;
}

float
epa_convolution(const std::vector<float>& sorted_coords
              , float h) {
  return epa_conv<float
                , float
                , BSIZE>(sorted_coords.data()
                       , sorted_coords.size()
                       , 1/h);
}

