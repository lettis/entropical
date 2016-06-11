
// WGSIZE set by host program

/* local probability from Epanechnikov kernel */
float
epanechnikov(float x
           , float ref_scaled_neg
           , float h_inv) {
  float p = fma(h_inv, x, ref_scaled_neg);
  p *= p;
  if (p <= 1.0f) {
    p = h_inv * fma(p, -0.75f, 0.75f);
  } else {
    p = 0.0f;
  }
  return p;
}

/* pre-reduce probabilities inside workgroup */
void
local_prob_reduction(uint lid
                   , __local float* p_wg
                   , __global float* P_partial) {
  uint stride;
  uint wid = get_group_id(0);
  // reduce locally
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      p_wg[lid] += p_wg[lid+stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    P_partial[wid] = p_wg[0] / ((float) WGSIZE);
  }
}


__kernel void
partial_probs_1d(__global const float* sorted_coords
               , unsigned int n_rows
               , __global float* P_partial
               , float h_inv
               , float ref_scaled_neg) {
//               , __constant float* h_inv
//               , __constant float* ref_scaled_neg) {
  __local float p_wg[WGSIZE];
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  // probability for every frame
  if (gid < n_rows) {
    p_wg[lid] = epanechnikov(sorted_coords[gid]
                           , ref_scaled_neg[0]
                           , h_inv[0]);
  } else {
    p_wg[lid] = 0.0f;
  }
  // pre-reduce inside workgroup
  local_prob_reduction(lid, p_wg, P_partial);
}

__kernel void
partial_probs_2d(__global const float* sorted_coords
               , unsigned int n_rows
               , __global float* P_partial
               , __constant float* h_inv
               , __constant float* ref_scaled_neg) {
  __local float p_wg[WGSIZE];
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  // probability for every frame
  if (gid < n_rows) {
    p_wg[lid] = epanechnikov(sorted_coords[gid]
                           , ref_scaled_neg[0]
                           , h_inv[0])
                * epanechnikov(sorted_coords[n_rows+gid]
                             , ref_scaled_neg[1]
                             , h_inv[1]);
  } else {
    p_wg[lid] = 0.0f;
  }
  // pre-reduce inside workgroup
  local_prob_reduction(lid, p_wg, P_partial);
}

__kernel void
partial_probs_3d(__global const float* sorted_coords
               , unsigned int n_rows
               , __global float* P_partial
               , __constant float* h_inv
               , __constant float* ref_scaled_neg) {
  __local float p_wg[WGSIZE];
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  // probability for every frame
  if (gid < n_rows) {
    p_wg[lid] = epanechnikov(sorted_coords[gid]
                   , ref_scaled_neg[0]
                   , h_inv[0])
                * epanechnikov(sorted_coords[n_rows+gid]
                             , ref_scaled_neg[1]
                             , h_inv[1])
                * epanechnikov(sorted_coords[2*n_rows+gid]
                             , ref_scaled_neg[2]
                             , h_inv[2]);
  } else {
    p_wg[lid] = 0.0f;
  }
  // pre-reduce inside workgroup
  local_prob_reduction(lid, p_wg, P_partial);
}


/* stagewise parallel reduction */
__kernel void
sum_partial_probs(__global float* P_partial
                , __global float* P
                , unsigned int i_ref
                , unsigned int n_partials
                , unsigned int n_wg
                , __global float* P_partial_reduct) {
  __local float p_wg[WGSIZE];
  uint stride;
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  // store probs locally for reduction
  if (gid < n_partials) {
    p_wg[lid] = P_partial[gid];
  } else {
    p_wg[lid] = 0.0f;
  }
  // reduce
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      p_wg[lid] += p_wg[lid+stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    if (get_num_groups(0) > 1) {
      // intermediate reduction result
      P_partial_reduct[wid] = p_wg[0];
    } else {
      // end result
      P[i_ref] = p_wg[0] / ((float) n_wg);
    }
  }
}

