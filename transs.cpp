
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <omp.h>

#include "coords_file/coords_file.hpp"
#include "tools.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    "\n\n"
    "compute transfer entropies between principal components"));
  opts.add_options()
    // required options
    ("input,i", po::value<std::string>()->required(), "principal components (required).")
    // optional parameters
    ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
    ("nthreads,n", po::value<unsigned int>()->default_value(0), "number of parallel threads (default: 0 == read from OMP_NUM_THREADS)")
    ("help,h", po::bool_switch()->default_value(false), "show this help.");
  try {
    po::store(po::command_line_parser(argc, argv).options(opts).run(), args);
    po::notify(args);
  } catch (po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments!\n\n";
    }
    std::cerr << opts << std::endl;
    return EXIT_FAILURE;
  }
  if (args["help"].as<bool>()) {
    std::cout << opts << std::endl;
    return EXIT_SUCCESS;
  }
  unsigned int nthreads = args["nthreads"].as<unsigned int>();
  if (nthreads > 0) {
    omp_set_num_threads(nthreads);
  }
  Tools::IO::set_out(args["output"].as<std::string>()); 
  std::string fname_input = args["input"].as<std::string>();

  //TODO open coords

  //TODO filter selected PCs

  // TODO probability definition over PCs
  //        (e.g. gaussian kernel density estimation per PC -> p(frame, PC))
  // TODO combined probability: P(x_n+1, x_n, y_n)
  // TODO combined probability: P(x_n+1, x_n)  -> P(x_n+1 | x_n) via Bayes

  // kernel density estimation:
  // f_h(x) = 1/(nh) \sum_i^n  K((x-x_i)/h)

  // gaussian kernel:
  //   K(x) = 1/sqrt(2pi) exp(-0.5 x^2)

  // Silverman's rule of thumb:
  //   h = 1.06 \sigma n^(-1/5)

  return EXIT_SUCCESS;
}

