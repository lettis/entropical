
// WGSIZE set by host program

/* local probability from Epanechnikov kernel */
float epanechnikov( float x
                  , float ref_scaled
                  , float h_inv_neg) {
  float p = fma(h_inv_neg, x, ref_scaled);
  p *= p;
  if (p <= 1.0f) {
    p = fma(p, -0.75f, 0.75f);
  }
  return p;
}

/* initialize buffer with zeros */
__kernel void initialize_zero(__global float* buf) {
  buf[get_global_id(0)] = 0.0f;
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
  __local float sum[WGSIZE];

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

  // P1: p(p_now, p_prev, p_tau)
  sum[lid] = p_now_wg[lid] * p_prev_wg[lid] * p_tau_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      sum[lid] += p_now_wg[lid+stride]
                  * p_prev_wg[lid+stride]
                  * p_tau_wg[lid+stride];
    }
  }
  if (lid == 0) {
    Ptmp.s0 = sum[0];
  }

  // P2: p(p_prev)
  sum[lid] = p_prev_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      sum[lid] += p_prev_wg[lid+stride];
    }
  }
  if (lid == 0) {
    Ptmp.s1 = sum[0];
  }

  // P3: p(p_prev, p_tau)
  sum[lid] = p_prev_wg[lid] * p_tau_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      sum[lid] += p_prev_wg[lid+stride]
                  * p_tau_wg[lid+stride];
    }
  }
  if (lid == 0) {
    Ptmp.s2 = sum[0];
  }

  // P4: p(p_now, p_prev)
  sum[lid] = p_now_wg[lid] * p_prev_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      sum[lid] += p_now_wg[lid+stride]
                  * p_prev_wg[lid+stride];
    }
  }
  if (lid == 0) {
    Ptmp.s3 = sum[0];
  }

  // store reduced probs for workgroup
  if (lid == 0) {
    Psingle[wid] = Ptmp;
  }
}


__kernel void collect_partials(__global const float4* Psingle
                             , __global float4* Pacc_partial
                             , __global float* Tacc_partial
                             , uint idx
                             , uint n
                             , uint n_workgroups) {

  uint i;
  float4 P_tmp = (float4) (0.0f);
  for (i=0; i < n_workgroups; ++i) {
    P_tmp += Psingle[i];
  }
  float T_tmp = 0.0f;
  if (P_tmp.s0 > 0.0f
   && P_tmp.s1 > 0.0f
   && P_tmp.s2 > 0.0f
   && P_tmp.s3 > 0.0f) {
    T_tmp = P_tmp.s0 * log2(P_tmp.s0*P_tmp.s1/P_tmp.s2/P_tmp.s3);
  }
  Pacc_partial[idx] = P_tmp;
  Tacc_partial[idx] = T_tmp;
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
      Pacc[lid] += Pacc_partial[lid+stride];
      Tacc[lid] += Tacc_partial[lid+stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // save workgroup result in global space
  if (lid == 0) {
    Pacc_partial[gid] = Pacc[0];
    Tacc_partial[gid] = Tacc[0];
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

