
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <functional>
#include <limits>
#include <cmath>

#include <boost/program_options.hpp>

#include <omp.h>

#include "entropical.hpp"

#include "tools.hpp"

#include "transs.hpp"
#include "mi.hpp"
#include "kldiv.hpp"
#include "dens.hpp"
#include "negs.hpp"
#include "hestimate.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  
  std::string general_help =
    "\n"
    "                /_'. _\n"
    "              _   \\ / '-.\n"
    "             < ``-.;),--'`\n"
    "              '--.</()`--.\n"
    "                / |/-/`'._\\\n"
    "  entropical    |/ |=|\n"
    "                   |_|\n"
    "              ~`   |-| ~~      ~\n"
    "          ~~  ~~ __|=|__   ~~\n"
    "        ~~   .-'`  |_|  ``""-._   ~~\n"
    "         ~~.'      |=|         '-.  ~\n"
    "           |      `\"\"\"`           \\   ~\n"
    "       ~   \\                      | ~~\n"
    "            '-.__.--._         .-'\n"
    "                 ~~   `--...-'`    ~\n"
    "\n\n"
    " compute entropy-based/information measures like\n"
    " local probabilities, mutual information, transfer entropies, etc.\n"
#ifdef USE_CUDA
    " code is accelerated using CUDA on Nvidia-GPUs.\n\n"
#else
    " code is accelerated using OpenMP on CPU(s).\n\n"
#endif
    "modes:\n"
    "  transs:    compute transfer entropies\n"
    "  kldiv:     compute Kullback-Leibler divergences\n"
    "  mi:        compute mutual informations\n"
    "  dens:      compute local probability densities in 1, 2 or 3 dimensions\n"
    "  negs:      compute negentropies\n"
    "  amise:     estimate bandwidths from AMISE-minimization\n"
    "  thumb:     estimate bandwidths from rule-of-thumb\n\n"
    "usage:\n"
    "  entropical MODE --option1 --option2 ...\n\n"
    "for a list of available options per mode, run with '-h' option, e.g.\n"
    "  entropical transs -h\n\n"
  ;
  enum Mode {TRANSS
           , KLDIV
           , MI
           , DENS
           , NEGS
           , AMISE
           , THUMB};
  std::map<std::string, Mode> mode_mapping {{"transs", TRANSS}
                                          , {"kldiv", KLDIV}
                                          , {"mi", MI}
                                          , {"dens", DENS}
                                          , {"negs", NEGS}
                                          , {"amise", AMISE}
                                          , {"thumb", THUMB}};
  Mode mode;
  if (argc <= 2) {
    std::cerr << general_help;
    return EXIT_FAILURE;
  } else {
    if (mode_mapping.count(argv[1]) == 0) {
      std::cerr << "\nerror: unrecognized mode '" << argv[1] << "'\n\n";
      std::cerr << general_help;
      return EXIT_FAILURE;
    } else {
      mode = mode_mapping[argv[1]];
    }
  }
  //// cmd arguments for all available modes
  po::options_description opts_common("");
  opts_common.add_options()
    ("help,h", po::bool_switch()->default_value(false),
      "show this help.")
    ("input,i", po::value<std::string>(),
      "coordinates (column-oriented space-separated plain text; required).")
    ("output,o", po::value<std::string>()->default_value(""),
      "output file (default: stdout)")
    ("columns,c", po::value<std::string>()->default_value(""),
      "column selection, space separated indices;\n"
      "e.g. -c \"1 3 5\" for first, third and fifth columns\n"
      "(default: read all)")
    ("verbose,v", po::bool_switch()->default_value(false),
      "verbose output.")
  ;
  // transs
  po::options_description opts_transs(
    "transs - compute transfer entropies between time series");
  opts_transs.add(opts_common);
  opts_transs.add_options()
    ("tau,t", po::value<unsigned int>()->default_value(1),
      "lagtime in # frames, default: 1")
    ("bandwidths,H", po::value<std::string>()->default_value(""),
      "bandwidths for univariate density estimation"
      " as space separated values.")
    ("bits,B", po::bool_switch()->default_value(false),
      "use log2 instead of ln")
  ;
  // kldiv
  po::options_description opts_kldiv(
    "kldiv - compute Kullback-Leibler divergences between time series");
  opts_kldiv.add(opts_common);
  opts_kldiv.add_options()
    ("bandwidths,H", po::value<std::string>()->default_value(""),
      "bandwidths for univariate density estimation"
      " as space separated values.")
    ("bits,B", po::bool_switch()->default_value(false),
      "use log2 instead of ln")
  ;
  // mi
  po::options_description opts_mi(
    "mi - compute mutual informations between time series");
  opts_mi.add(opts_common);
  opts_mi.add_options()
    ("bandwidths,H", po::value<std::string>()->default_value(""),
      "bandwidths for univariate density estimation"
      " as space separated values.")
    ("bits,B", po::bool_switch()->default_value(false),
      "use log2 instead of ln")
  ;
  // dens
  po::options_description opts_dens(
    "dens - compute local probability densities of time series");
  opts_dens.add(opts_common);
  opts_dens.add_options()
    ("bandwidths,H", po::value<std::string>()->default_value(""),
      "bandwidths for univariate density estimation"
      " as space separated values.")
    ("taus", po::value<std::string>()->default_value(""),
      "lag times for probability density estimation "
      "(must be equal to # of dimensions; default: no time lag.)")
    ("dims,D", po::value<unsigned int>()->default_value(1),
      "kernel dimensionality for multivariate density estimation (dens)")
  ;
  // negs
  po::options_description opts_negs(
    "negs - compute negentropies of time series");
  opts_negs.add(opts_common);
  opts_negs.add_options()
    ("bandwidths,H", po::value<std::string>()->default_value(""),
      "bandwidths for univariate density estimation"
      " as space separated values.")
    ("bits,B", po::bool_switch()->default_value(false),
      "use log2 instead of ln")
  ;
  // amise
  po::options_description opts_amise(
    "amise - estimate probability kernel bandwidths via amise-minimization");
  opts_amise.add(opts_common);
//  opts_amise.add_options()
//    ("bandwidths,H", po::value<std::string>()->default_value(""),
//      "bandwidths for univariate density estimation"
//      " as space separated values.")
  ;
  // thumb
  po::options_description opts_thumb(
    "thumb - estimate probability kernel bandwidths via rule-of-thumb");
  opts_thumb.add(opts_common);

  // option parsing
  po::variables_map args;
  std::map<Mode, po::options_description> opts {
    {TRANSS, opts_transs}
  , {KLDIV, opts_kldiv}
  , {MI, opts_mi}
  , {DENS, opts_dens}
  , {NEGS, opts_negs}
  , {AMISE, opts_amise}
  , {THUMB, opts_thumb}};
  try {
    po::store(po::command_line_parser(argc, argv)
                .options(opts[mode])
                .run()
            , args);
    po::notify(args);
    // documentation
    if (args["help"].as<bool>()) {
      std::cout << opts[mode] << std::endl;
      return EXIT_SUCCESS;
    }
    // select mode ...
    std::map<Mode, std::function<void(po::variables_map)>> subroutines;
    subroutines[TRANSS] = Transs::main;
    subroutines[MI] = Mi::main;
    subroutines[KLDIV] = KLDiv::main;
    subroutines[DENS] = Dens::main;
    subroutines[NEGS] = Negs::main;
    subroutines[AMISE] = Hestimate::AmiseMin::main;
    subroutines[THUMB] = Hestimate::Thumb::main;
    // ... and run corresponding subroutine
    subroutines[mode](args);
  } catch (const boost::bad_any_cast& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments!\n\n";
      std::cerr << opts[mode] << std::endl;
    }
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "\n" << e.what() << "\n\n";
    std::cerr << opts[mode] << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

