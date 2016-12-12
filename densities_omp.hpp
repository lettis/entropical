#pragma once

#include <vector>

namespace Densities {
namespace OMP {
  
  struct epanechnikov {
    float operator() (float ref_val
                    , float val
                    , float h) const;
  };

}} // end namespace Densities::OMP

/**
 * Perform combined density computation for 1-3 observables.
 */
std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<unsigned int> i_cols
                 , std::vector<float> h);

/**
 * Perform combined density computation for 1-3 observables with time lags.
 */
std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<unsigned int> i_cols
                 , std::vector<float> h
                 , std::vector<unsigned int> tau);


/**
 * Compute convolution of Epanechnikov kernel for AMISE estimation
 */
float
epa_convolution(const std::vector<float>& sorted_coords
              , float h);

