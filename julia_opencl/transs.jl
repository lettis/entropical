#!/usr/bin/env julia

import ArgParse
import OpenCL
const cl = OpenCL

function gpu_platform()
  # screen all platforms for gpu devices
  for  p in cl.platforms()
    if cl.available_devices(p)[1][:device_type] == :gpu
      return p
    end
  end
end

function tranfer_entropy(X, i, j)

end


function main()
  args_settings = ArgParse.ArgParseSettings()
  ArgParse.@add_arg_table args_settings begin
    "--input", "-i"
      help = "input file."
    "--output", "-o"
      help = "output file."
    "--tau", "-t"
      arg_type = Int
      default = 0
      help = "lagtime (in # frames; default: 1"
    "--maxcol"
      arg_type = Int
      default = 0
      help = "number of highest column to read as observable. (default: all)"
  end
  args = ArgParse.parse_args(args_settings)

  gpus = gpu_platform()
  if typeof(gpus) == Void
    println("no gpus!")
  else
    @printf("%d gpus found!\n", length(cl.available_devices(gpus)))
  end





#########################
  #TODO read data
  #TODO loop over i,j pairs
  #         compute T_{ij} and T_{ji} in one call
  #TODO output
end

main()

