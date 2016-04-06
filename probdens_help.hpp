
#pragma once

#include <map>
#include <string>

namespace Probdens {
  const std::map<std::string, std::string> help = {
    {"help-transs",
      ""}

  , {"help-mi",
      ""}

  , {"help-dens",
      ""}

  , {"help-negs",
      ""}

  , {"help-amise",
      ""}

  , {"help-hestimate",
      "\nhestimate - estimate kernel bandwidths for kernel density estimation\n"
      "\n"
      "estimation is done by Silverman's rule of thumb\n"
      "\n"
      "options:\n"
      "  -i [ --input ]   input file (required!)\n"
      "  -o [ --output ]  output file (default: stdout)\n"
      "  -c [ --col ]     column for which bandwidth should be estimated\n"
      "                   (default: 0 = estimate for all columns)\n"
    }
  };

  //TODO: write documentation
  

  /**
   * Prints extensive documentation for given mode.
   */
  void
  print_help_and_exit(std::string helpmode);

} // end namespace 'Probdens'

