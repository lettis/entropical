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
      default = 32
      help = "workgroup size for OpenCL kernels; default: 32"
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
      #define WGSIZE %d

      /* probabilities from Epanechnikov kernel */
      __kernel void probs(__global const float* data
                        , __global float* p
                        , float ref_scaled
                        , float h_inv_neg
                        , uint n) {
        uint gid = get_global_id[0];
        float p_tmp = 0.0f;
        if (gid < n) {
          p_tmp = fma(h_inv_neg, data[gid], ref_scaled);
          p_tmp *= p_tmp;
          if (p_tmp <= 1) {
            p_tmp = fma(p_tmp, -0.75, 0.75);
          }
        }
        p[gid] = p_tmp;
      }

//TODO: combine both kernel into one map-reduce step

      /* reduce probabilities to partial product-kernel sums
         by stagewise-pairwise parallel summation */
      __kernel void reduce_probs(__global const float* p_now
                               , __global const float* p_prev
                               , __global const float* p_tau
                               , __global float* P1partial
                               , __global float* P2partial
                               , __global float* P3partial
                               , __global float* P4partial) {
        __local float p_now_wg[WGSIZE];
        __local float p_prev_wg[WGSIZE];
        __local float p_tau_wg[WGSIZE];
        __local float sum[WGSIZE];

        uint stride;
        uint gid = get_global_id(0);
        uint lid = get_local_id(0);
        uint wid = get_group_id(0);

        p_now_wg[lid] = p_now[gid];
        p_prev_wg[lid] = p_prev[gid];
        p_tau_wg[lid] = p_tau[gid];
        barrier(CLK_LOCAL_MEM_FENCE);

        /* P1: p(p_now, p_prev, p_tau) */
        sum[lid] = p_now_wg[lid] * p_prev_wg[lid] * p_tau_wg[lid];
        for (stride=WGSIZE/2; stride > 0; stride /= 2) {
          barrier(CLK_LOCAL_MEM_FENCE);
          if (local_id < stride) {
            sum[lid] += p_now_wg[lid+stride]
                        * p_prev_wg[lid+stride]
                        * p_tau_wg[lid+stride];
          }
        }
        if (lid == 0) {
          P1partial[wid] = sum[0];
        }

        /* P2: p(p_prev) */
        sum[lid] = p_prev_wg[lid];
        for (stride=WGSIZE/2; stride > 0; stride /= 2) {
          barrier(CLK_LOCAL_MEM_FENCE);
          if (local_id < stride) {
            sum[lid] += p_prev_wg[lid+stride];
          }
        }
        if (lid == 0) {
          P2partial[wid] = sum[0];
        }

        /* P3: p(p_prev, p_tau) */
        sum[lid] = p_prev_wg[lid] * p_tau_wg[lid];
        for (stride=WGSIZE/2; stride > 0; stride /= 2) {
          barrier(CLK_LOCAL_MEM_FENCE);
          if (local_id < stride) {
            sum[lid] += p_prev_wg[lid+stride]
                        * p_tau_wg[lid+stride];
          }
        }
        if (lid == 0) {
          P3partial[wid] = sum[0];
        }

        /* P4: p(p_now, p_prev) */
        sum[lid] = p_now_wg[lid] * p_prev_wg[lid];
        for (stride=WGSIZE/2; stride > 0; stride /= 2) {
          barrier(CLK_LOCAL_MEM_FENCE);
          if (local_id < stride) {
            sum[lid] += p_now_wg[lid+stride]
                        * p_prev_wg[lid+stride];
          }
        }
        if (lid == 0) {
          P4partial[wid] = sum[0];
        }
      }


      __kernel void collect_partials(const float* P1partial
                                   , const float* P2partial
                                   , const float* P3partial
                                   , const float* P4partial
                                   , float* Pacc
                                   , float* T
                                   , uint idx_T
                                   , uint n_workgroups) {
        float P1 = 0.0f;
        float P2 = 0.0f;
        float P3 = 0.0f;
        float P4 = 0.0f;
        uint i;
        for (i=0; i < n_workgroups; ++i) {
          P1 += P1partial[i];
          P2 += P2partial[i];
          P3 += P3partial[i];
          P4 += P4partial[i];
        }
        Pacc[0] += P1;
        Pacc[1] += P2;
        Pacc[2] += P3;
        Pacc[3] += P4;
        T[idx_T] += P1 * log2(P1*P2/P3/P4);
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
      queue = cl.CmdQueue(ctx, :profile)
      prg = cl.Program(ctx, source=@sprintf(kernel_src, wgsize)) |> cl.build!
      # setup kernels
      probs_krnl = cl.Kernel(prg, "probs")
      reduce_probs_krnl = cl.Kernel(prg, "reduce_probs")
      # setup buffers
      i_buf = cl.Buffer(Float32, ctx, :r, n_extended)
      j_buf = cl.Buffer(Float32, ctx, :r, n_extended)
      p_tau_buf = cl.Buffer(Float32, ctx, :rw, n_extended)
      p_prev_buf = cl.Buffer(Float32, ctx, :rw, n_extended)
      p_buf = cl.Buffer(Float32, ctx, :rw, n_extended)
      p1_partial_buf = cl.Buffer(Float32, ctx, :rw, n_workgroups)
      p2_partial_buf = cl.Buffer(Float32, ctx, :rw, n_workgroups)
      p3_partial_buf = cl.Buffer(Float32, ctx, :rw, n_workgroups)
      p4_partial_buf = cl.Buffer(Float32, ctx, :rw, n_workgroups)
      Pacc_buf = cl.Buffer(Float32, ctx, :rw, 4)
      T_buf = cl.Buffer(Float32, ctx, :rw, length(partitioning))

      function kernel_invocation(buf1, buf2, ii, jj)
        for k in tau:n
          #TODO blocking vs. non-blocking kernel calls
          #     (cl.call vs cl.enqueue_kernel)
          probs_krnl[queue, (n_extended), wgsize]
                      (buf1
                     , p_tau_buf
                     , data[k-tau,ii]/bandwidths[ii]
                     , 1/bandwidths[ii]
                     , n)

          probs_krnl[queue, (n_extended), wgsize]
                      (buf2
                     , p_buf
                     , data[k,jj]/bandwidths[jj]
                     , 1/bandwidths[jj]
                     , n)

          probs_krnl[queue, (n_extended), wgsize]
                      (buf2
                     , p_prev_buf
                     , data[k-1,jj]/bandwidths[jj]
                     , 1/bandwidths[jj]
                     , n)

          reduce_probs_krnl[queue, (n_extended), wgsize]
                      (p_buf
                     , p_prev_buf
                     , p_tau_buf
                     , p1_partial_buf
                     , p2_partial_buf
                     , p3_partial_buf
                     , p4_partial_buf)

          collect_partials_krnl[queue, 1, 1]
                      (p1_partial_buf
                     , p2_partial_buf
                     , p3_partial_buf
                     , p4_partial_buf
                     , Pacc_buf
                     , T_buf
                     , idx
                     , n_workgroups)
        end
        #TODO retrieve T from GPU
      end

      # compute transfer entropies on this gpu
      for (idx, (i,j)) in enumerate(partitioning)
        cl.write!(queue, i_buf, data[:,i])
        cl.write!(queue, j_buf, data[:,j])
        kernel_invocation(i_buf, j_buf, i, j)
        kernel_invocation(j_buf, i_buf, j, i)
        #TODO write to T
      end

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

