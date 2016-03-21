#!/usr/bin/env julia

import ArgParse
import OpenCL
const cl = OpenCL
import Iterators

# screen all platforms for gpu devices and return the right one
function gpu_platform()
  for  p in cl.platforms()
    if cl.available_devices(p)[1][:device_type] == :gpu
      return p
    end
  end
end

# upper triangle of pairwise indices
upperΔ(n) = sort(filter(x->x[1]<x[2], collect(Iterators.product(1:n,1:n))))

# produce equal load partition indices for given number of gpus
function load_partitions(ndims, ngpus)
  partitions = map(_->[], 1:ngpus)
  pairs = sort(upperΔ(ndims), rev=true)
  i = pairs[end][1]
  function assign_to_gpus(order)
    for g in order
      while length(pairs) > 0 && pairs[end][1] == i
        push!(partitions[g], pop!(pairs))
      end
      if length(pairs) > 0
        i = pairs[end][1]
      end
    end
  end
  while length(pairs) > 0
    assign_to_gpus(1:ngpus)
    assign_to_gpus(ngpus:-1:1)
  end
  return partitions
end



function main()
  args_settings = ArgParse.ArgParseSettings()
  ArgParse.@add_arg_table args_settings begin
    "--input", "-i"
      help = "input file."
    "--output", "-o"
      help = "output file."
    "--ncols", "-c"
      arg_type = Int
      help = "number of highest column to read as observable."
    "--tau", "-t"
      arg_type = Int
      default = 0
      help = "lagtime (in # frames; default: 1"
    "--wgsize"
      arg_type = Int
      default = 64
      help = "workgroup size for OpenCL kernels; default: 64; if changed,
              make it a multiple of 64 for optimal performance (says at least
              AMD)"
  end
  args = ArgParse.parse_args(args_settings)

  # check if gpus are available
  gpus = gpu_platform()
  ngpus = 0
  if typeof(gpus) == Void
    println("error: no gpus found!")
    exit(1)
  else
    ngpus = length(cl.available_devices(gpus))
    @printf("running on %d gpus\n", ngpus)
  end
  # add enough worker nodes to feed the gpus
  addprocs(ngpus)

  # prepare code for use in parallel workers
  @everywhere begin
    import OpenCL
    cl = OpenCL

    function load_data(fname, ncols)
      data = readdlm(fname, Float32)
      if ncols < size(data)[2]
        data = data[:,1:ncols]
      end
      return data
    end

    function silvermans_rule(data)
      (nrows, ncols) = size(data)
      bandwidths = zeros(ncols)
      n_scaled = 1.06 * nrows^(-0.2)
      for i in 1:ncols
        bandwidths[i] = std(data[:,i]) * n_scaled
      end
      return bandwidths
    end

    kernel_src = "
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
    "

    # wrapper for parallel processing
    function worker(fname, ncols, partitioning, wgsize)
      data = load_data(fname, ncols)
      #TODO better bandwidth selection via Jones & Sheather?
      #     https://www.jstor.org/stable/2345597
      bandwidths = silvermans_rule(data)
      # extend with empty rows to align memory to group size
      n = size(data)[1]
      n_workgroups = cld(n, wgsize)
      n_extended = n_workgroups * wgsize
      data = [data ; zeros(n_extended - n, ncols)]
      # setup OpenCL environment for this worker
      gpu_platform = 0
      for p in cl.platforms()
        if cl.available_devices(p)[1][:device_type] == :gpu
          gpu_platform = p
          break
        end
      end
      gpu = cl.available_devices(gpu_platform)[myid()-1]
      ctx = cl.Context(gpu)
      #TODO: profile flag?
      queue = cl.CmdQueue(ctx, :profile)
      prg = cl.Program(ctx, source="#define WGSIZE $wgsize\n$kernel_src")
      prg = cl.build!(prg, raise=false)
      

      #### setup kernels
      # compute partially reduced probabilities
      partial_probs_krnl = cl.Kernel(prg, "partial_probs")
      # compute fully reduced probabilities / frame
      collect_partials_krnl = cl.Kernel(prg, "collect_partials")
      # accumulate T (i.e. produce final result)
      compute_T_krnl = cl.Kernel(prg, "compute_T")

      #### setup buffers
      # buffers for input data
      i_buf = cl.Buffer(Float32, ctx, :r, n_extended)
      j_buf = cl.Buffer(Float32, ctx, :r, n_extended)
      # buffer for temporary (partially reduced) probabilities
      Psingle_buf = cl.Buffer(Float32, ctx, :rw, 4*n_workgroups)
      # buffer for fully reduced probabilities (per frame)
      Pacc_partial_buf = cl.Buffer(Float32, ctx, :rw, 4*n_extended)
      # buffer for temporary transfer entropies (per frame)
      T_partial_buf = cl.Buffer(Float32, ctx, :rw, n_extended)
      # buffer for transfer entropies (final results)
      T_buf = cl.Buffer(Float32, ctx, :rw, length(partitioning))

      # helper function to run kernels for transfer entropy
      # in specified direction (ii -> jj)
      function kernel_invocation(buf1, buf2, i, j, idx)
        for k in tau:n
          cl.set_args!(partial_probs_krnl
                     , buf1
                     , buf2
                     , float32(data[k    ,j]/bandwidths[j])
                     , float32(data[k-1  ,j]/bandwidths[j])
                     , float32(data[k-tau,i]/bandwidths[i])
                     , float32(-1/bandwidths[i])
                     , float32(-1/bandwidths[j])
                     , Psingle_buf)
          cl.enqueue_kernel(queue
                          , partial_probs_krnl
                          , n_extended
                          , wgsize)

          cl.set_args!(collect_partials_krnl
                     , Psingle_buf
                     , Pacc_partial_buf
                     , uint32(k)
                     , uint32(n)
                     , uint32(n_workgroups))
          cl.enqueue_task(queue
                        , collect_partials_krnl)
        end

        cl.set_args!(compute_T_krnl
                   , Pacc_partial_buf
                   , T_partial_buf
                   , n
                   , n_workgroups
                   , T_buf
                   , idx)
        cl.enqueue_kernel(queue
                        , compute_T_krnl
                        , n_extended
                        , wgsize)
      end

      # compute transfer entropies on this gpu
      last_i = 0
      for (idx, (i,j)) in enumerate(partitioning)
        if i != last_i
          # {ij}-pairs are ordered by i, so we
          # can save some copies to the GPU by
          # checking if the i-dimension has not changed
          cl.write!(queue, i_buf, data[:,i])
          last_i = i
        end
        cl.write!(queue, j_buf, data[:,j])
        # compute T_{ij}
        kernel_invocation(i_buf, j_buf, i, j, idx)
        # compute T_{ji}
        kernel_invocation(j_buf, i_buf, j, i, idx)
      end

      cl.wait()
      T = cl.read(queue, T_buf)
      return T
    end
  end # @everywhere

  # partition indices to balanced load for gpus
  load_partitioning = load_partitions(args["ncols"], ngpus)
  # parallel computation on all available gpus
  results = cell(ngpus)
  @sync begin
    for i in 1:ngpus
      @async results[i] = remotecall_fetch(worker
                                         , i+1
                                         , args["input"]
                                         , args["ncols"]
                                         , load_partitioning[i]
                                         , args["wgsize"])
    end
  end
  #TODO write results to output
  for r in results
    println(r)
  end
end

main()

