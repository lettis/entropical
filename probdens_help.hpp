#pragma once

#include <map>
#include <string>

namespace Probdens {
  std::map<std::string, std::string> help;

  //TODO: write documentation
  
  help["help-transs"] = "";

  help["help-mi"] = "";

  help["help-dens"] = "";

  help["help-amise"] = "";

  help["help-hestimate"] = "";

  /**
   * Prints extensive documentation for given mode.
   */
  void
  print_help_and_exit(std::string helpmode);

} // end namespace 'Probdens'

