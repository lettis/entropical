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

    kernel_src = "
      __kernel void transs(__global const float* i
                         , __global const float* j
                         , __global float* result) {
     }
    "
    
    # wrapper for parallel processing
    function worker(fname, ncols, partition)
      data = load_data(fname, ncols)
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
      reduce_krnl = cl.Kernel(prg, "reduce")
      #global_size = (n,)
      #local_size =  (nworkers,)

      # compute transfer entropies on this gpu
      T = Dict()
      for (i,j) in sort(partition)
        i_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=data[:,i])
        j_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=data[:,j])

        # event = kern[queue, global_size, local_size](i_buf, j_buf, ...)
        # event[:profile_duration]  # [s]

        #TODO run kernels
        #TODO reduce results
      end
      return T
    end
  end # @everyvwhere

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
                                         , load_partitioning[i])
    end
  end
  #TODO write results to output
  for r in results
    println(r)
  end
end

main()

