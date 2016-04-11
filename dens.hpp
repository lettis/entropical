
#pragma once

#include <boost/program_options.hpp>

namespace Dens {

  /**
   * Creates context, queue and compiled kernels for GPU and
   * allocates buffers needed for local density estimation.
   */
  void
  setup_gpu(GPUElement& gpu
          , std::string kernel_src
          , unsigned int n_rows);

  /**
   * Check command line options, read input,
   * run probability density estimation and
   * write output to file (or stdout).
   */
  void
  main(boost::program_options::variables_map args);

} // end namespace 'Dens'

