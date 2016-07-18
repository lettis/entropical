
#include "densities_cuda.hpp"
#include "tools.hpp"

#include "density_1d.cuh"


std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<std::size_t> i_cols
                 , std::vector<float> h
                 , std::vector<std::size_t> tau) {
  std::size_t n_dim = i_cols.size();
  if (n_dim < 1 || 3 < n_dim) {
    std::cerr << "error: can only compute combined probabilities in 1, 2 or 3 "
              << "dimensions."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (h.size() != n_dim) {
    std::cerr << "error: number of bandwidth parameters does not match number "
              << "of selected columns."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (tau.size() != n_dim) {
    std::cerr << "error: number of lagtimes (tau) does not match number of "
              << "selected dimensions."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // prepare data
  std::vector<float> h_inv(n_dim);
  for (std::size_t n=0; n < n_dim; ++n) {
    h_inv[n] = 1.0f/h[n];
  }
  // create filtered coords (row-major order)
  // from original coords (col-major order)
  std::vector<float> sel_coords(n_rows*n_dim);
  for (unsigned int i=0; i < n_rows; ++i) {
    for (unsigned int j=0; j < n_dim; ++j) {
      sel_coords[i*n_dim+j] = coords[i_cols[j]*n_rows+i];
    }
  }
  switch(n_dim) {
    1:
      return density_1d(sel_coords.data()
                      , n_rows
                      , 1
                      , h_inv);
    2:
      //TODO implement
      return {};
    3:
      //TODO implement
      return {};
    default:
      std::cerr << "error: in # of dim! this should never happen!" <<std::endl;
      exit(EXIT_FAILURE);
  }
}

