
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <set>
#include <omp.h>

#include "coords_file/coords_file.hpp"
#include "tools.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS] -i FILE [PC1 [PC2 ...]]\n\n"
    "compute transfer entropies between principal components"));
  opts.add_options()
//TODO add PC ids as positional arguments
    // required options
    ("input,i", po::value<std::string>()->required(), "principal components (required).")
    // optional parameters
    ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
    ("covmatrix,C", po::value<std::string>(), "PCA covariance matrix (for PC sigmas)")
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
  //TODO: get PC ids from args
  std::set<std::size_t> pcs;

  std::vector<double> sigmas;
  if (args.count("covmatrix")) {
    // load sigmas from covariance matrix
    CoordsFile::FilePointer fh = CoordsFile::open(fname_input, "r");
    std::size_t i=1;
    while ( ! fh->eof()) {
      //TODO consider if PC should be read
      std::vector<float> line = fh->next();
      if (line.size() > 0) {
        // sigma_i = sqrt(cov_{ii})
        sigmas.push_back(sqrt(line[i]));
        ++i;
      }
    }
  } else {
    // no covariance matrix available: compute sigmas from data
    //TODO
  }

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

