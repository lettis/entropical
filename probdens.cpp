
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <omp.h>

#include "probdens.hpp"
#include "probdens_help.hpp"

#include "tools.hpp"

#include "transs.hpp"




int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS] FILE\n\n"
    "compute local probabilities and information transfer"));
  try {
    opts.add_options()
      ("input,i", po::value<std::string>()->required(), "principal components (required).")
      ("mode,m", po::value<std::string>()->required(),
        "one of: transs (transfer entropies),"
        "        mi (mutual information),"
        "        dens (local density),"
        "        amise (AMISE),"
        "        hestimate (bandwidth estimation for kernel density)")

      ("col,c", po::value<unsigned int>()->default_value(0), "column for univariate densities")
      ("colmax,C", po::value<unsigned int>()->default_value(0), "max. column (default: 0 == read all)")

      


      ("tau,t", po::value<unsigned int>()->default_value(1), "lagtime for transfer entropy calculation (in # frames; default: 1).")
      ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
      ("wgsize", po::value<unsigned int>()->default_value(64), "workgroup size (default: 64)")
      ("verbose,v", po::bool_switch()->default_value(false), "verbose output.")
      
      ("help-transs", po::bool_switch()->default_value(false), "detailed help for 'transs' mode.");
      ("help-mi", po::bool_switch()->default_value(false), "detailed help for 'mi' mode.");
      ("help-dens", po::bool_switch()->default_value(false), "detailed help for 'dens' mode.");
      ("help-amise", po::bool_switch()->default_value(false), "detailed help for 'amise' mode.");
      ("help-hestimate", po::bool_switch()->default_value(false), "detailed help for 'hestimate' mode.");
      ("help,h", po::bool_switch()->default_value(false), "show this help.");

    // option parsing
    po::positional_options_description pos_opts;
    pos_opts.add("input", 1);
    po::store(po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run(), args);
    po::notify(args);
    // documentation
    auto check_special_help = [&] (std::string mode) -> void {
      if (args[mode].as<bool>()) {
        Probdens::print_help_and_exit(mode);
      }
    };
    if (args["help"].as<bool>()) {
      std::cout << opts << std::endl;
      return EXIT_SUCCESS;
    }
    check_special_help("help-transs");
    check_special_help("help-mi");
    check_special_help("help-dens");
    check_special_help("help-amise");
    check_special_help("help-hestimate");
    // input, settings, checks
    bool verbose = args["verbose"].as<bool>();
    unsigned int wgsize = args["wgsize"].as<unsigned int>();
    // set output stream to file or STDOUT, depending on args
    Tools::IO::set_out(args["output"].as<std::string>()); 
    std::string fname_input = args["input"].as<std::string>();
    unsigned int pc_max = args["pcmax"].as<unsigned int>();
    if (pc_max == 1) {
      std::cerr << "error: need at least two PCs to compute information transfer" << std::endl;
      return EXIT_FAILURE;
    }
    unsigned int tau = args["tau"].as<unsigned int>();
    if (tau == 0) {
      std::cerr << "error: a lagtime of 0 frames does not make sense." << std::endl;
      return EXIT_FAILURE;
    }
    // read coordinates
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    if (pc_max == 0) {
      verbose && std::cout << "reading full dataset" << std::endl;
      // read full data set
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C');
    } else {
      // read only until given column
      verbose && std::cout << "reading dataset up to PC " << pc_max << std::endl;
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C', Tools::range<std::size_t>(0, pc_max, 1));
    }

    if (mode == "transs") {
      std::vector<std::vector<float>> T = compute_transfer_entropies(coords
                                                                   , n_rows
                                                                   , n_cols
                                                                   , tau
                                                                   , {}
                                                                   , wgsize);
      // output
      for (std::size_t y=0; y < n_cols; ++y) {
        for (std::size_t x=0; x < n_cols; ++x) {
          Tools::IO::out() << " " << T[y][x];
        }
        Tools::IO::out() << "\n";
      }
    } else {
      std::cerr << "error: unknown mode '" << mode << "'" << std::endl;
    }

    // memory cleanup
    Tools::IO::free_coords(coords);
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

