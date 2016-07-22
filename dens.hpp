#pragma once

#include <vector>
#include <boost/program_options.hpp>

namespace Dens {

  /**
   * Compute 1D-densities for given observables.
   * Runs in parallel on all available GPUs.
   */
  std::vector<std::vector<float>>
  compute_densities_1d(std::vector<std::size_t> selected_cols
                     , const float* coords
                     , std::size_t n_rows
                     , std::vector<float> bandwidths);

  /**
   * Compute 2D/3D-densities for given observables.
   */
  std::tuple<std::vector<std::vector<float>>, std::vector<std::string>>
  compute_densities_nd(std::vector<std::size_t> selected_cols
                     , const float* coords
                     , std::size_t n_rows
                     , std::vector<float> bandwidths
                     , unsigned int dim_kernel
                     , std::vector<unsigned int> taus);

  /**
   * Check command line options, read input,
   * run probability density estimation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);

} // end namespace 'Dens'

