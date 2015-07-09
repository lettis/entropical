
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <vector>
#include <algorithm>
#include <memory>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <omp.h>

#include "tools.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS] -i FILE PC1 PC2 [PC3 ...]\n\n"
    "compute transfer entropies between principal components"));
  try {
    opts.add_options()
      // required options
      ("input,i", po::value<std::string>()->required(), "principal components (required).")
      // optional parameters
      ("pcmax", po::value<unsigned int>()->default_value(-1), "max. PC to read (default: 0 == read all)")
      ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
      ("nthreads,n", po::value<unsigned int>()->default_value(0), "number of parallel threads (default: 0 == read from OMP_NUM_THREADS)")
      ("help,h", po::bool_switch()->default_value(false), "show this help.");
    // option parsing, settings, checks
    po::positional_options_description pos_opts;
    pos_opts.add("pc-ids", -1);
    po::store(po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run(), args);
    po::notify(args);
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
    unsigned int pc_max = args["pc_max"].as<unsigned int>();
    if (pc_max == 1) {
      std::cerr << "error: need at least two PCs to compute information transfer" << std::endl;
      return EXIT_FAILURE;
    }




    // read frames of principal components and compute sigmas for every dimension
//    std::vector<double> sigmas;
//    std::vector<std::vector<double>> frames;
//    std::size_t n_frames = 0;
//    {
//      using namespace boost::accumulators;
//      using VarAcc = accumulator_set<double, features<tag::variance(lazy)>>;
//      std::vector<std::shared_ptr<VarAcc>> acc(n_pcs);
//      for (std::size_t i=0; i < n_pcs; ++i) {
//        acc[i] = std::shared_ptr<VarAcc>(new VarAcc);
//      }
//      CoordsFile::FilePointer fh = CoordsFile::open(fname_input, "r");
//      for (std::size_t pc: pcs) {
//        frames[pc] = {};
//      }
//      while ( ! fh->eof()) {
//        std::vector<float> frame = fh->next();
//        if (frame.size() > 0) {
//          ++n_frames;
//          for (std::size_t pc: pcs) {
//            (*acc[pc-1])(frame[pc-1]);
//            frames[pc].push_back(frame[pc-1]);
//          }
//        }
//      }
//      // collect resulting sigmas from accumulators
//      for (std::size_t i=0; i < n_pcs; ++i) {
//        sigmas[i] = sqrt(variance((*acc[i])));
//      }
//    }

    //TODO normalized N-dim histogram from PDFs
    // for every bin:
    //    compute local prob density (estimate pdf at bin center)
    // normalize grid

    //TODO entropy from (histograms of) multivariate PDFs
    // S = \sum p log(p)


    // TODO probability definition over PCs
    //        (e.g. gaussian kernel density estimation per PC -> p(frame, PC))
   
    // TODO combined probability: P(x_n+1, x_n, y_n)
    //        3D Gaussian Kernel
    
    // TODO combined probability: P(x_n+1, x_n)
    //        2D Gaussian Kernel or Bayes from 1D Gaussian of P(y_n) and P(x_n+1, x_n, y_n)

    // multivariate product kernel:
    //  kernel_density_estimation.pdf : p25

  } catch (const boost::bad_any_cast& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments!\n\n";
    }
    std::cerr << opts << std::endl;
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "\n" << e.what() << "\n\n";
    std::cerr << opts << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

