#pragma once

#include <vector>

namespace Densities {
namespace OMP {
  
  struct epanechnikov {
    float operator() (float ref_val
                    , float val
                    , float h) const;
  };

  struct epanechnikov_var {
    float operator() (float ref_val
                    , float val
                    , float h) const;
  };

  struct epanechnikov_bias {
    float operator() (float ref_val
                    , float val
                    , float h) const;
  };

  //TODO: implement
  template <typename KernelFunctor>
  std::vector<float>
  per_frame_1d(const float* coords
             , std::vector<unsigned int> i_col
             , const std::vector<float>& sorted_coords
             , std::vector<float> h);

}} // end namespace Densities::OMP

/**
 * Perform combined density computation for 1-3 observables on CPU.
 */
std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<unsigned int> i_cols
                 , std::vector<float> h);

/**
 * Perform combined density computation for 1-3 observables.
 */
std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<unsigned int> i_cols
                 , std::vector<float> h
                 , std::vector<unsigned int> tau);

