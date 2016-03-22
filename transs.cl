
/* local probability from Epanechnikov kernel */
float epanechnikov( float x
                  , float ref_scaled
                  , float h_inv_neg) {
  float p = fma(h_inv_neg, x, ref_scaled);
  p *= p;
  if (p_tmp <= 1.0f) {
    p = fma(p, -0.75f, 0.75f);
  }
  return p;
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
                          , __global float4* Psingle) {
  __local float p_now_wg[WGSIZE];
  __local float p_prev_wg[WGSIZE];
  __local float p_tau_wg[WGSIZE];
  __local float sum[WGSIZE];

  uint stride;
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);

  float4 Ptmp;

  //TODO: better performance if put inside if-clause?
  float from = buf_from[gid];
  float to = buf_to[gid];

  if (gid < n) {
    p_now_wg[lid] = epanechnikov(to, ref_now_scaled, h_inv_neg_2);
    p_prev_wg[lid] = epanechnikov(to, ref_prev_scaled, h_inv_neg_2);
    p_tau_wg[lid] = epanechnikov(from, ref_tau_scaled, h_inv_neg_1);
  } else {
    p_now_wg[lid] = 0.0f;
    p_prev_wg[lid] = 0.0f;
    p_tau_wg[lid] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* P1: p(p_now, p_prev, p_tau) */
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
    Ptmp[0] = sum[0];
  }

  /* P2: p(p_prev) */
  sum[lid] = p_prev_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      sum[lid] += p_prev_wg[lid+stride];
    }
  }
  if (lid == 0) {
    Ptmp[1] = sum[0];
  }

  /* P3: p(p_prev, p_tau) */
  sum[lid] = p_prev_wg[lid] * p_tau_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      sum[lid] += p_prev_wg[lid+stride]
                  * p_tau_wg[lid+stride];
    }
  }
  if (lid == 0) {
    Ptmp[2] = sum[0];
  }

  /* P4: p(p_now, p_prev) */
  sum[lid] = p_now_wg[lid] * p_prev_wg[lid];
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      sum[lid] += p_now_wg[lid+stride]
                  * p_prev_wg[lid+stride];
    }
  }
  if (lid == 0) {
    Ptmp[3] = sum[0];
  }

  Psingle[wid] = Ptmp;
}


__kernel void collect_partials(const float4* Psingle
                             , float4* Pacc_partial
                             , float* T_partial
                             , uint idx
                             , uint n
                             , uint n_workgroups) {
  uint i;
  float4 P_tmp = (float4) (0.0f);
  for (i=0; i < n_workgroups; ++i) {
    P_tmp += Psingle[i];
  }
  float T_tmp = P_tmp[0] * log2(P_tmp[0]*P_tmp[1]/P_tmp[2]/P_tmp[3]);
  Pacc_partial[idx] = P_tmp;
  T_partial[idx] = T_tmp;
}


__kernel void compute_T(float* Pacc_partial
                      , float* T_partial
                      , uint n
                      , uint n_workgroups
                      , float* T
                      , uint idx) {
  __local float4 Pacc[WGSIZE];
  __local float Tacc[WGSIZE];
  float4 P = (float4) (0.0f);
  float Ttmp = 0.0f;

  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint wid = get_group_id(0);
  uint stride, i;

  /* copy data to local memory */
  if (gid < n) {
    Pacc[lid] = Pacc_partial[gid];
    Tacc[lid] = T_partial[gid];
  } else {
    Pacc[lid] = (float4)(0.0f);
    Tacc[lid] = 0.0f;
  }

  /* parallel reduction in workgroups */
  for (stride=WGSIZE/2; stride > 0; stride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
      Pacc[lid] += Pacc_partial[lid+stride];
      Tacc[lid] += T_partial[lid+stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* save workgroup result in global space */
  if (lid == 0) {
    Pacc_partial[gid] = Pacc[0];
    T_partial[gid] = Tacc[0];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  /* compute T from reduced results */
  if (gid == 0) {
    for (i=0; i < n_workgroups; ++i) {
      Pacc[i] = Pacc_partial[i];
      Tacc[i] = T_partial[i];
    }
    for (i=0; i < n_workgroups; ++i) {
      P += Pacc[i];
      Ttmp += Tacc[i];
    }

    /* renormalize T by total probs P */
    Ttmp = 1/P[0] * (Ttmp + log2(P[2]*P[3]/P[0]/P[1]));

    /* write result to global buffer */
    T[idx] = Ttmp;
  }
}

