
#pragma once

#include <vector>
#include <boost/program_options.hpp>

#include "tools_opencl.hpp"

namespace Dens {

  /**
   * Perform stagewise parallel reduction of probabilities directly on GPU.
   */
  void
  stagewise_reduction(Tools::OCL::GPUElement* gpu
                    , unsigned int i_ref
                    , unsigned int n_partials
                    , unsigned int n_wg
                    , std::size_t wgsize);

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
  std::vector<float>
  compute_densities_1d(Tools::OCL::GPUElement* gpu
                     , const float* coords
                     , std::size_t n_rows
                     , std::size_t i_col_1
                     , std::size_t i_col_2
                     , float h1
                     , float h2
                     , std::size_t n_wg
                     , std::size_t wgsize);

  /**
   * Compute densities for given observables.
   * Runs in parallel on all available GPUs.
   */
  std::vector<std::vector<float>>
  compute_densities(std::vector<std::size_t> selected_cols
                  , const float* coords
                  , std::size_t n_rows
                  , std::vector<float> bandwidths);

  /**
   * Check command line options, read input,
   * run probability density estimation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);

} // end namespace 'Dens'

