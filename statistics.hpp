#pragma once

#include <vector>
#include <utility>

namespace Statistics {

//TODO: documentation
struct Histogram {
  Histogram();

  // prefix
  Histogram& operator++();
  // postfix
  Histogram operator++(int);


};

// TODO: documentation.
// pair: first begin, second end
Histogram
histogram(std::vector<std::size_t> resolution
        , std::vector<double> bandwidths
        , std::vector<std::vector<double>> data
        , std::vector<std::size_t> begin_configuration = {}
        , std::vector<std::size_t> end_configuration = {});

double
entropy(Histogram hist);

} // end namespace Statistics

