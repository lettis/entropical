
#pragma once

#include <map>
#include <string>

namespace Probdens {
  /**
   * Detailed documentation on different modes.
   */
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
      "Estimation is done by Silverman's rule of thumb.\n"
      //TODO reference
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

