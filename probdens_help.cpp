
#include "probdens_help.hpp"

namespace Probdens {

  void
  print_help_and_exit(std::string helpmode) {
    auto is_mode = [helpmode] (std::string scomp) -> bool {
      return (scomp.compare(helpmode) == 0);
    };

    if (is_mode("help-transs")
     || is_mode("help-mi")
     || is_mode("help-dens")
     || is_mode("help-amise")
     || is_mode("help-hestimate")) {
      std::cout << help[helpmode] << std::endl;
      exit(EXIT_SUCCESS);
    } else {
      std::cerr << "no help available for option '"
                << helpmode << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

} // end namespace 'Probdens'

