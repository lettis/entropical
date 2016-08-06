#pragma once

namespace IM {
  // flags for (joint) probabilities
  enum {X, X_XTAU, Y, Y_YTAU, X_Y, X_XTAU_Y, Y_YTAU_X, N_PROBS};
  // flags for transfer entropies
  enum {XY, YX, N_T};

  /**
   * Host code for transfer entropy calculation.
   * Detects available GPUs and parallelizes job on these.
   */
  std::vector<std::vector<float>> 
  compute_transfer_entropies(const float* coords
                           , std::size_t n_rows
                           , std::size_t n_cols
                           , unsigned int tau
                           , std::vector<float> bandwidths
                           , unsigned int wgsize);

} // end namespace Transs

