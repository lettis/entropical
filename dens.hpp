
#pragma once

#include <vector>
#include <boost/program_options.hpp>

#include "tools_opencl.hpp"

namespace Dens {

  /**
   * Perform density computation on given GPU.
   */
  std::vector<float>
  compute_densities(Tools::OCL::GPUElement& gpu
                  , float* coords
                  , std::size_t n_rows
                  , std::size_t i_col
                  , const std::vector<float>& sorted_coords
                  , float h
                  , std::size_t n_wg
                  , std::size_t wgsize);

  /**
   * Check command line options, read input,
   * run probability density estimation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);

} // end namespace 'Dens'

