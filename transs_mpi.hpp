#pragma once

#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <omp.h>
#include <mpi.h>

#include "tools.hpp"

// define process 0 as main process
const int MAIN_PROCESS = 0;

int main_mpi(int argc, char* argv[]);

