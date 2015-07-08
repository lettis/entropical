
#include "statistics.hpp"

namespace Statistics {

Histogram
histogram(std::vector<std::size_t> resolution
        , std::vector<double> bandwidths
        , std::vector<std::vector<double>> data
        , std::vector<std::size_t> initial_configuration = {}) {
  //TODO
}

double
entropy(Histogram hist) {
  // TODO
}

} // end namespace Statistics

