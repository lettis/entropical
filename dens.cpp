
#include <omp.h>

#include "dens.hpp"
#include "tools.hpp"

namespace Dens {

  std::vector<float>
  compute_densities(Tools::OCL::GPUElement& gpu
                  , float* coords
                  , std::size_t n_rows
                  , std::size_t n_cols
                  , std::size_t i_col) {
    std::vector<float> densities(n_rows);

    //TODO compute densities
    
    return densities;
  }

  void
  main(boost::program_options::variables_map args) {
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    // read data
    std::vector<std::size_t> selected_cols;
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::tie(selected_cols, coords, n_rows, n_cols)
      = Tools::IO::selected_coords<float>(fname_input
                                        , args["columns"].as<std::string>());
    // setup OpenCL environment
    std::vector<Tools::OCL::GPUElement> gpus = Tools::OCL::gpus();
    std::size_t n_gpus = gpus.size();
    if (n_gpus == 0) {
      Tools::OCL::print_no_gpus_errmsg();
      exit(EXIT_FAILURE);
    }
    //TODO finish OpenCL setup
    // compute local densities on GPUs
    unsigned int i, j, thread_id;
    unsigned int n_selected_cols = selected_cols.size();
    std::vector<std::vector<float>> densities(selected_cols.size());
    #pragma omp parallel for default(none)\
                             private(j,thread_id)\
                             firstprivate(n_selected_cols,n_rows,n_cols)\
                             shared(coords,selected_cols,gpus,densities)\
                             schedule(dynamic,1)
    for (j=0; j < n_selected_cols; ++j) {
      thread_id = omp_get_thread_num();
      densities[j] = compute_densities(gpus[thread_id]
                                     , coords
                                     , n_rows
                                     , n_cols
                                     , selected_cols[j]);
    }
    // output: densities
    for (i=0; i < n_rows; ++i) {
      for (j=0; j < selected_cols.size(); ++j) {
        Tools::IO::out() << " " << densities[j][i];
      }
      Tools::IO::out() << "\n";
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Dens'

