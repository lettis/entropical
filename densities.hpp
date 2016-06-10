#pragma once

#include "tools.hpp"
#include "tools_opencl.hpp"


/**
 * Prepare GPU for 1D density computation.
 * @returns number of workgroups needed for computation
 */
unsigned int
prepare_gpus_1d(std::vector<Tools::OCL::GPUElement>& gpus
              , unsigned int wgsize
              , std::size_t n_rows);

/**
 * Perform density computation for one observable on given GPU.
 */
std::vector<float>
compute_densities_1d(Tools::OCL::GPUElement* gpu
                   , const float* coords
                   , std::size_t n_rows
                   , std::size_t i_col
                   , float h
                   , std::size_t n_wg
                   , std::size_t wgsize);

/**
 * Perform combined density computation for two observables on given GPU.
 */
//std::vector<float>
//compute_densities_2d(Tools::OCL::GPUElement* gpu
//                   , const float* coords
//                   , std::size_t n_rows
//                   , std::size_t i_col[2]
//                   , float h[2]
//                   , std::size_t n_wg
//                   , std::size_t wgsize_2d);

