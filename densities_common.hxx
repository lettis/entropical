
#include "densities_common.hpp"

#include <iostream>
#include <algorithm>

template <typename NUM>
std::vector<NUM>
prob_dens_coord_prep(const NUM* coords
                   , std::size_t n_rows
                   , std::vector<unsigned int> i_cols
                   , std::vector<NUM> h
                   , std::vector<unsigned int> tau
                   , bool row_major_result) {
  unsigned int n_dim = i_cols.size();
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
  // create filtered coords (row-major order)
  // from original coords (col-major order)
  // honoring lagtimes tau
  unsigned int tau_max = (*std::max_element(tau.begin(), tau.end()));
  unsigned int n_rows_sel = n_rows - tau_max;
  std::vector<NUM> sel_coords(n_rows_sel*n_dim);
  for (unsigned int i=0; i < n_rows_sel; ++i) {
    for (unsigned int j=0; j < n_dim; ++j) {
      if (row_major_result) {
        sel_coords[i*n_dim+j] = coords[i_cols[j]*n_rows+i+tau[j]];
      } else {
        sel_coords[j*n_rows_sel+i] = coords[i_cols[j]*n_rows+i+tau[j]];
      }
    }
  }
  return sel_coords;
}

