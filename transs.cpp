
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <vector>
#include <algorithm>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <omp.h>

#include "coords_file/coords_file.hpp"
#include "tools.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS] -i FILE PC1 PC2 [PC3 ...]\n\n"
    "compute transfer entropies between principal components"));
  opts.add_options()
    // required options
    ("input,i", po::value<std::string>()->required(), "principal components (required).")
    ("pc-ids", po::value<std::vector<unsigned int>>(), "PC ids")
    // optional parameters
    ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
    ("covmatrix,C", po::value<std::string>(), "PCA covariance matrix (for PC sigmas)")
    ("nthreads,n", po::value<unsigned int>()->default_value(0), "number of parallel threads (default: 0 == read from OMP_NUM_THREADS)")
    ("help,h", po::bool_switch()->default_value(false), "show this help.");
  // option parsing, settings, checks
  po::positional_options_description pos_opts;
  pos_opts.add("pc-ids", -1);
  try {
    po::store(po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run(), args);
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
  std::vector<unsigned int> pcs = args["pc-ids"].as<std::vector<unsigned int>>();
  std::size_t n_pcs = pcs.size();
  if (n_pcs < 2) {
    std::cerr << "error: need at least two PCs to compute information transfer" << std::endl;
    return EXIT_FAILURE;
  } else {
    std::sort(pcs.begin(), pcs.end());
  }

  // read/compute sigmas and N_frames for bandwidth (h) selection
  std::vector<double> sigmas(n_pcs);
  std::size_t n_frames=0;
  if (args.count("covmatrix")) {
    // load sigmas from covariance matrix
    {
      CoordsFile::FilePointer fh = CoordsFile::open(args["covmatrix"].as<std::string>(), "r");
      // list principal components in base-1.
      std::size_t pc=1;
      std::size_t i_next_pc=0;
      while ( ! fh->eof()) {
        std::vector<float> line = fh->next();
        if (line.size() > 0) {
          if (pc == pcs[i_next_pc]) {
            // sigma_i = sqrt(cov_{ii})
            sigmas.push_back(sqrt(line[pc-1]));
            ++i_next_pc;
            if (i_next_pc >= n_pcs) {
              break;
            }
          }
          ++pc;
        }
      }
    }
    // count N_frames
    {
      CoordsFile::FilePointer fh = CoordsFile::open(fname_input, "r");
      while ( ! fh->eof()) {
        std::vector<float> frame = fh->next();
        if (frame.size() > 0) {
          ++n_frames;
        }
      }
    }
  } else {
    // no covariance matrix available: compute sigmas from principal components directly
    using namespace boost::accumulators;
    using VarAcc = accumulator_set<double, features<tag::variance(lazy)>>;
    CoordsFile::FilePointer fh = CoordsFile::open(fname_input, "r");
    std::vector<VarAcc> acc(n_pcs);
    while ( ! fh->eof()) {
      std::vector<float> frame = fh->next();
      if (frame.size() > 0) {
        ++n_frames;
        for (std::size_t i=0; i < n_pcs; ++i) {
          acc[i](frame[pcs[i]-1]);
        }
      }
    }
    // collect results
    for (std::size_t i=0; i < n_pcs; ++i) {
      sigmas[i] = sqrt(variance(acc[i]));
    }
  }

  // bandwidth selection based on
  // Silverman's rule of thumb:
  //   h = 1.06 \sigma n^(-1/5)
  //
  //
  // TODO use Scott's rule for multi-dimensional spaces!
  //
  std::vector<double> h;
  double n_frames_scaled = std::pow(n_frames, -0.2);
  for (double sigma: sigmas) {
    h.push_back(1.06 * sigma * n_frames_scaled);
  }



  // TODO probability definition over PCs
  //        (e.g. gaussian kernel density estimation per PC -> p(frame, PC))
  // TODO combined probability: P(x_n+1, x_n, y_n)
  // TODO combined probability: P(x_n+1, x_n)  -> P(x_n+1 | x_n) via Bayes

  // kernel density estimation:
  // f_h(x) = 1/(nh) \sum_i^n  K((x-x_i)/h)

  // gaussian kernel:
  //   K(x) = 1/sqrt(2pi) exp(-0.5 x^2)


  return EXIT_SUCCESS;
}

