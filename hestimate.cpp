
#include "hestimate.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>

namespace Hestimate {

  void
  main(boost::program_options::variables_map args) {
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    unsigned int col = args["col"].as<unsigned int>();

    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    if (col == 0) {
      // read all columns
      std::tie(coords, n_rows, n_cols) =
          Tools::IO::read_coords<float>(fname_input, 'C');
    } else {
      // read only specified column
      std::tie(coords, n_rows, n_cols) =
          Tools::IO::read_coords<float>(fname_input, 'C', {col-1});
    }
    // estimate bandwidth(s)
    for (std::size_t i_col=0; i_col < n_cols; ++i) {
      Tools::IO::out() << " " << silverman(coords, n_rows, i_col);
    }
    Tools::IO::out() << std::endl;
    free(coords);
  }

  float
  silverman(float* coords
          , std::size_t n_rows
          , std::size_t i_col) {
    using namespace boost::accumulators;
    using VarAcc = accumulator_set<float, features<tag::variance(lazy)>>;
    VarAcc acc;
    for (std::size_t i=0; i < n_rows; ++i) {
      acc(coords[col*n_rows+i]);
    }
    return std::pow(n_rows, -1.0/7.0)*sqrt(variance(acc));
  }

} // end namespace 'Hestimate'

