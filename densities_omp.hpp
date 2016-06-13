#pragma once

#include <vector>

/**
 * Perform combined density computation for 1-3 observables on CPU.
 */
std::vector<float>
combined_densities(const float* coords
                 , std::size_t n_rows
                 , std::vector<std::size_t> i_col
                 , std::vector<float> h);

