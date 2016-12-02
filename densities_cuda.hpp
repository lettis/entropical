#pragma once

#include <vector>


/**
 * Perform combined density computation for 1-3 observables.
 * tau defaults to zero.
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

