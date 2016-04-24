
// WGSIZE set by host program

/* local probability from Epanechnikov kernel */
float epanechnikov( float x
                  , float ref_scaled
                  , float h_inv_neg) {
  float p = fma(h_inv_neg, x, ref_scaled);
  p *= p;
  if (p <= 1.0f) {
    p = fma(p, -0.75f, 0.75f);
  } else {
    p = 0.0f;
  }
  return p;
}

/* initialize buffer with zeros */
__kernel void initialize_zero(__global float* buf) {
  buf[get_global_id(0)] = 0.0f;
}

__kernel void
probs_1d(__global const float* coords
       , float h_inv_neg
       , float ref_scaled
       , __global float* P_partial
       , unsigned int n
       , __global float* P) {
  __local float p_wg[WGSIZE];
//TODO: this code works only, if coords are sorted!
  uint stride;
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint n_wg = get_num_groups(0);
  // probability for every frame
  if (gid < n) {
    p_wg[lid] = h_inv_neg * epanechnikov(coords[gid], ref_scaled, h_inv_neg);
  } else {
    p_wg[lid] = 0.0f;
  }
  // reduce locally
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      p_wg[lid] += p_wg[lid+stride];
    }
  }
  // reduce globally
  if (lid == 0) {
    P_partial[wid] = -1.0f / ((float) WGSIZE) * p_wg[0];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (gid == 0) {
    float Pacc = 0.0f;
    for (uint i=0; i < n_wg; ++i) {
      Pacc += P_partial[i];
    }
    P[n] = Pacc / ((float) n_wg);
  }
}

/* compute and reduce probabilities to partial product-kernel sums
   by stagewise-pairwise parallel summation */
__kernel void partial_probs(__global const float* buf_from
                          , __global const float* buf_to
                          , float ref_now_scaled
                          , float ref_prev_scaled
                          , float ref_tau_scaled
                          , float h_inv_neg_1
                          , float h_inv_neg_2
                          , __global float4* Psingle
                          , unsigned int n) {
  __local float p_now_wg[WGSIZE];
  __local float p_prev_wg[WGSIZE];
  __local float p_tau_wg[WGSIZE];
  __local float sum1[WGSIZE];
  __local float sum2[WGSIZE];
  __local float sum3[WGSIZE];
  __local float sum4[WGSIZE];

  uint stride;
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);

  float4 Ptmp;

  if (gid < n) {
    float from = buf_from[gid];
    float to = buf_to[gid];
    p_now_wg[lid] = epanechnikov(to, ref_now_scaled, h_inv_neg_2);
    p_prev_wg[lid] = epanechnikov(to, ref_prev_scaled, h_inv_neg_2);
    p_tau_wg[lid] = epanechnikov(from, ref_tau_scaled, h_inv_neg_1);
  } else {
    p_now_wg[lid] = 0.0f;
    p_prev_wg[lid] = 0.0f;
    p_tau_wg[lid] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  sum1[lid] = p_now_wg[lid] * p_prev_wg[lid] * p_tau_wg[lid];
  sum2[lid] = p_prev_wg[lid];
  sum3[lid] = p_prev_wg[lid] * p_tau_wg[lid];
  sum4[lid] = p_now_wg[lid] * p_prev_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      float p_now_tmp = p_now_wg[lid+stride];
      float p_prev_tmp = p_prev_wg[lid+stride];
      float p_tau_tmp = p_tau_wg[lid+stride];
      // P2: p(p_prev)
      sum2[lid] += p_prev_tmp;
      // P3: p(p_prev, p_tau)
      sum3[lid] += p_prev_tmp
                   * p_tau_tmp;
      // P1: p(p_now, p_prev, p_tau)
      sum1[lid] += p_now_tmp
                   * p_prev_tmp
                   * p_tau_tmp;
      // P4: p(p_now, p_prev)
      sum4[lid] += p_now_tmp
                   * p_prev_tmp;
    }
  }
  if (lid == 0) {
    Ptmp.s0 = sum1[0];
    Ptmp.s1 = sum2[0];
    Ptmp.s2 = sum3[0];
    Ptmp.s3 = sum4[0];
    Psingle[wid] = Ptmp;
  }
}


__kernel void collect_partials(__global float4* Psingle
                             , __global float4* Pacc_partial
                             , __global float* Tacc_partial
                             , uint idx
                             , uint n
                             , uint n_workgroups) {
  float4 P_tmp[WGSIZE];
  uint stride;
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  // init local memory
  if (gid < n_workgroups) {
    P_tmp[lid] = Psingle[gid];
  } else {
    P_tmp[lid] = (float4) (0.0f);
  }
  // parallel reduction inside workgroups
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      P_tmp[lid] += P_tmp[lid+stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // save workgroup result in global memory
  // reusing Psingle as buffer
  if (lid == 0) {
    Psingle[wid] = P_tmp[0];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // collect workgroup results
  if (gid == 0) {
    uint i;
    float4 P = (float4) (0.0f);
    for (i=0; i < get_num_groups(0); ++i) {
      P += Psingle[i];
    }

    float T_tmp = 0.0f;
    if (P.s0 > 0.0f
     && P.s1 > 0.0f
     && P.s2 > 0.0f
     && P.s3 > 0.0f) {
      T_tmp = P.s0 * log2(P.s0*P.s1/P.s2/P.s3);
    }
    Pacc_partial[idx] = P;
    Tacc_partial[idx] = T_tmp;
  }
}

__kernel void compute_T(__global float4* Pacc_partial
                      , __global float* Tacc_partial
                      , uint n
                      , uint n_workgroups
                      , __global float* T
                      , uint idx) {
  __local float4 Pacc[WGSIZE];
  __local float Tacc[WGSIZE];
  float4 P = (float4) (0.0f);
  float Ttmp = 0.0f;

  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint stride, i;

  // copy data to local memory
  if (gid < n) {
    Pacc[lid] = Pacc_partial[gid];
    Tacc[lid] = Tacc_partial[gid];
  } else {
    Pacc[lid] = (float4)(0.0f);
    Tacc[lid] = 0.0f;
  }

  // parallel reduction in workgroups
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      Pacc[lid] += Pacc[lid+stride];
      Tacc[lid] += Tacc[lid+stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // save workgroup result in global space
  // (reusing Pacc_partial as temporary buffer)
  if (lid == 0) {
    Pacc_partial[wid] = Pacc[0];
    Tacc_partial[wid] = Tacc[0];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // compute T from reduced results
  if (gid == 0) {
    for (i=0; i < n_workgroups; ++i) {
      P += Pacc_partial[i];
      Ttmp += Tacc_partial[i];
    }

    // renormalize T by total probs P
    if (P.s0 > 0.0f
     && P.s1 > 0.0f
     && P.s2 > 0.0f
     && P.s3 > 0.0f) {
      Ttmp = 1.0f/P.s0 * (Ttmp + log2(P.s2*P.s3/P.s0/P.s1));
    }

    // write result to global buffer
    T[idx] = Ttmp;
  }
}

