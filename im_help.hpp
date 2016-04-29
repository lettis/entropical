
#pragma once

#include <map>
#include <string>

namespace IM {
  /**
   * Detailed documentation on different modes.
   */
  const std::map<std::string, std::string> help = {
    {"help-transs",
      ""}

  , {"help-mi",
      ""}

  , {"help-dens",
      "dens - compute local probability densities using 1D kernel density estimation\n"
      "\n"
      "options:\n"
      "  -i [ --input ]      input file (required!)\n"
      "  -H [ --bandwidths ] bandwidths for univariate density estimation\n"
      "                      as space separated values. (required!)\n"
      "  -o [ --output ]     output file (default: stdout)\n"
      "  -C [ --columns ]    column selection, space separated indices;\n"
      "                      e.g. -C \"1 3 5\" for first, third and fifth columns\n"
      "                      (default: read all)\n"
    }

  , {"help-negs",
      "negs - compute negentropies for given data columns\n"
      "\n"
      "options:\n"
      "  -i [ --input ]      input file (required!)\n"
      "  -H [ --bandwidths ] bandwidths for univariate density estimation\n"
      "                      as space separated values. (required!)\n"
      "  -o [ --output ]     output file (default: stdout)\n"
      "  -C [ --columns ]    column selection, space separated indices;\n"
      "                      e.g. -C \"1 3 5\" for first, third and fifth columns\n"
      "                      (default: read all)\n"
    }

  , {"help-amise",
      ""}

  , {"help-hestimate",
      "\nhestimate - estimate kernel bandwidths for kernel density estimation\n"
      "\n"
      "Estimation is done by Silverman's rule of thumb\n"
      "(see: B.W. Silverman, \"Density Estimation for Statistics and Data"
      " Analysis\".\n London: Chapman & Hall/CRC. p. 48. ISBN 0-412-24620-1.)\n"
      "\n"
      "options:\n"
      "  -i [ --input ]   input file (required!)\n"
      "  -o [ --output ]  output file (default: stdout)\n"
      "  -c [ --col ]     column for which bandwidth should be estimated\n"
      "                   (default: 0 = estimate for all columns)\n"
    }
  };

  /**
   * Prints extensive documentation for given mode and exits program.
   */
  void
  print_help_and_exit(std::string helpmode);

} // end namespace 'Probdens'

