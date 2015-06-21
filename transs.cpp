
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <omp.h>

#include "coords_file/coords_file.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    "\n\n"
    "compute transfer entropies between principal components."));
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
  std::string fname_input = args["input"].as<std::string>();
  std::string fname_output = args["output"].as<std::string>();
  std::ofstream ofs;
  if (fname_output != "") {
    ofs.open(fname_output);
  }
  std::ostream& out = (fname_output != "" ? ofs : std::cout);




  return EXIT_SUCCESS;
}

