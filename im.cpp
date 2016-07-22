
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <functional>
#include <limits>
#include <cmath>

#include <boost/program_options.hpp>

#include <omp.h>

#include "im.hpp"
#include "im_help.hpp"

#include "tools.hpp"

#include "transs.hpp"
#include "mi.hpp"
#include "dens.hpp"
#include "negs.hpp"
#include "amise.hpp"
#include "hestimate.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS]\n\n"
    "compute information measures like local probabilities,"
    " mutual information, information transfer, etc.\n"
#ifdef USE_CUDA
    "code is accelerated by CUDA on Nvidia-GPUs.\n\n"
#else
    "code is accelerated by OpenMP on CPU.\n\n"
#endif
    "options"));
  try {
    opts.add_options()
      ("input,i", po::value<std::string>(),
        "coordinates (column-oriented space-separated plain text; required).")
      ("output,o", po::value<std::string>()->default_value(""),
        "output file (default: stdout)")

      ("mode,m", po::value<std::string>(),
        "one of:\n"
        "        transs (transfer entropies),\n"
        "        mi (mutual information),\n"
        "        dens (local probability density),\n"
        "        negs (negentropy),\n"
        "        amise (AMISE),\n"
        "        hestimate (bandwidth estimation)")

      ("columns,c", po::value<std::string>()->default_value(""),
        "column selection, space separated indices;\n"
        "e.g. -c \"1 3 5\" for first, third and fifth columns\n"
        "(default: read all)")
      ("bandwidths,H", po::value<std::string>()->default_value(""),
        "bandwidths for univariate density estimation"
        " as space separated values.")
      ("dims,D", po::value<unsigned int>()->default_value(1),
        "kernel dimensionality for multivariate density estimation (dens)")
      ("taus,T", po::value<std::string>()->default_value(""),
        "lagtimes for combined density estimation, "
        "as list with (positive!) value per dimension (compare --columns/-c) "
        "(in # frames, default: no lag)")

      ("verbose,v", po::bool_switch()->default_value(false),
        "verbose output.")

      ("help-transs", po::bool_switch()->default_value(false),
        "detailed help for 'transs' mode.")
      ("help-mi", po::bool_switch()->default_value(false),
        "detailed help for 'mi' mode.")
      ("help-dens", po::bool_switch()->default_value(false),
        "detailed help for 'dens' mode.")
      ("help-negs", po::bool_switch()->default_value(false),
        "detailed help for 'negs' mode.")
      ("help-amise", po::bool_switch()->default_value(false),
        "detailed help for 'amise' mode.")
      ("help-hestimate", po::bool_switch()->default_value(false),
        "detailed help for 'hestimate' mode.")
      ("help,h", po::bool_switch()->default_value(false),
        "show this help.")
      ;

    // option parsing
    po::store(po::command_line_parser(argc, argv).options(opts).run(), args);
    po::notify(args);
    // documentation
    auto check_special_help = [&] (std::string helpmode) -> void {
      if (args[helpmode].as<bool>()) {
        IM::print_help_and_exit(helpmode);
      }
    };
    if (args["help"].as<bool>()) {
      std::cout << opts << std::endl;
      return EXIT_SUCCESS;
    }
    check_special_help("help-transs");
    check_special_help("help-mi");
    check_special_help("help-dens");
    check_special_help("help-negs");
    check_special_help("help-amise");
    check_special_help("help-hestimate");
    // select mode and run corresponding subroutine
    std::string mode = args["mode"].as<std::string>();
    std::map<std::string, std::function<void(po::variables_map)>> subroutines;
    subroutines["transs"] = Transs::main; //TODO
    subroutines["mi"] = Mi::main; //TODO
    subroutines["dens"] = Dens::main;
    subroutines["negs"] = Negs::main; // TODO
    subroutines["amise"] = Amise::main; // TODO
    subroutines["hestimate"] = Hestimate::main;
    if (subroutines.count(mode) == 0) {
      std::cerr << "error: unknown mode '" << mode << "'" << std::endl;
    } else {
      subroutines[mode](args);
    }

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

