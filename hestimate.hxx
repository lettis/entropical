
#include "tools.hpp"

namespace Hestimate {

  template <typename KRNL>
  void
  main(boost::program_options::variables_map args) {
    KRNL krnl;
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    std::vector<std::size_t> selected_cols;
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::tie(selected_cols, coords, n_rows, n_cols)
      = Tools::IO::selected_coords<float>(fname_input
                                        , args["columns"].as<std::string>());
    // write bandwidth estimation to 'out'
    for (std::size_t i_col=0; i_col < n_cols; ++i_col) {
      Tools::IO::out() << " " << krnl(coords, n_rows, i_col);
    }
    Tools::IO::out() << std::endl;
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end Hestimate

