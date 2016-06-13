#pragma once

#include "tools.hpp"
#include "tools_opencl.hpp"


/**
 * Prepare GPU for 1D density computation.
 * @returns number of workgroups needed for computation
 */
unsigned int
prepare_gpus(std::vector<Tools::OCL::GPUElement>& gpus
           , unsigned int wgsize
           , std::size_t n_rows
           , std::size_t n_dim);

/**
 * Perform combined density computation for 1-3 observables on given GPU.
 */
std::vector<float>
combined_densities(Tools::OCL::GPUElement* gpu
                 , const float* coords
                 , std::size_t n_rows
                 , std::vector<std::size_t> i_col
                 , std::vector<float> h
                 , std::size_t n_wg
                 , std::size_t wgsize);

