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
      /* probabilities from Epanechnikov kernel */
      __kernel void probs(__global const float* data
                        , __global float* p
                        , float ref_scaled
                        , float h_inv_neg
                        , int n) {
        int gid = get_global_id[0];
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
    "

    # wrapper for parallel processing
    function worker(fname, ncols, partition, wgsize)
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
      prg = cl.Program(ctx, source=kernel_src) |> cl.build!
      # setup kernels
      probs_krnl = cl.Kernel(prg, "probs")
      # setup buffers
      i_buf = cl.Buffer(Float32, ctx, :r, n_extended)
      j_buf = cl.Buffer(Float32, ctx, :r, n_extended)
      p_tau_buf = cl.Buffer(Float32, ctx, :rw, n_extended)
      p_prev_buf = cl.Buffer(Float32, ctx, :rw, n_extended)
      p_buf = cl.Buffer(Float32, ctx, :rw, n_extended)
      # compute transfer entropies on this gpu
      T = Dict()
      for (i,j) in sort(partition)
        cl.write!(queue, i_buf, data[:,i])
        cl.write!(queue, j_buf, data[:,j])
        # compute kernel evaluations per k for different prob-values
        for k in tau:n
          probs_krnl[queue, (n_extended), wgsize]
                      (i_buf
                     , p_tau_buf
                     , data[k-tau,i]/bandwidths[i]
                     , 1/bandwidths[i]
                     , n)
          probs_krnl[queue, (n_extended), wgsize]
                      (j_buf
                     , p_buf
                     , data[k,j]/bandwidths[j]
                     , 1/bandwidths[j]
                     , n)
          probs_krnl[queue, (n_extended), wgsize]
                      (j_buf
                     , p_prev_buf
                     , data[k-1,j]/bandwidths[j]
                     , 1/bandwidths[j]
                     , n)
          # TODO: reduce kernel evaluations, i.e., compute product kernel


          # TODO: the same, vice versa (i <-> j)


        end

        # TODO: reduce prob-values to T_ij

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

