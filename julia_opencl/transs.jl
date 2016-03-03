#!/usr/bin/env julia

import argparse
import OpenCL
const cl = OpenCL

function main()
  args_settings = argparse.ArgParseSettings()
  argparse.@add_arg_table args_settings begin
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
  args = argparse.parse_args(args_settings)

  ### setup OpenCL

  # get devices
  # create queue per device








end





