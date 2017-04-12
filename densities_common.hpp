#pragma once

#include <vector>

/**
 * assert correct dimensionality and according number of tau and h values.
 *
 * @returns selected coords (in row- or column-major order)
 *          honoring tau values.
 */
template <typename NUM>
std::vector<NUM>
prob_dens_coord_prep(const NUM* coords
                   , std::size_t n_rows
                   , std::vector<unsigned int> i_cols
                   , std::vector<NUM> h
                   , std::vector<unsigned int> tau
                   , bool row_major_result);

////

#include "densities_common.hxx"

