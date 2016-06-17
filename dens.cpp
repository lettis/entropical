
#include <algorithm>
#include <string>
#include <omp.h>
#include <vector>

#include "dens.hpp"

#ifdef USE_OPENCL
  #include "tools_opencl.hpp"
  #include "densities_opencl.hpp"
#elif USE_CUDA
  #include "densities_cuda.hpp"
#else
  #include "densities_omp.hpp"
#endif

#include "tools.hpp"

namespace Dens {

  std::vector<std::vector<float>>
  compute_densities_1d(std::vector<std::size_t> selected_cols
                     , const float* coords
                     , std::size_t n_rows
                     , std::vector<float> bandwidths) {
    std::vector<std::vector<float>> densities(selected_cols.size());
    unsigned int n_selected_cols = selected_cols.size();
    unsigned int j;
#ifdef USE_OPENCL
    //// compute probability densities with OpenCL on GPU(s)
    // setup OpenCL environment
    std::vector<Tools::OCL::GPUElement> gpus = Tools::OCL::gpus();
    std::size_t n_gpus = gpus.size();
    if (n_gpus == 0) {
      Tools::OCL::print_no_gpus_errmsg();
      exit(EXIT_FAILURE);
    }
    // TODO: workgroup size is hard-coded to optimum for GeForce GTX-960.
    //       this has to be changed to a relative value of max(wgsize).
    unsigned int wgsize1d = 128;
    // compute densities on available GPUs
    unsigned int n_wg = prepare_gpus(gpus
                                   , wgsize1d
                                   , n_rows
                                   , 1);
    unsigned int thread_id;
    #pragma omp parallel for default(none)\
                             private(j,thread_id)\
                             firstprivate(n_selected_cols,n_rows,\
                                          n_wg,wgsize1d)\
                             shared(coords,selected_cols,gpus,\
                                    densities,bandwidths)\
                             num_threads(n_gpus)\
                             schedule(dynamic,1)
    for (j=0; j < n_selected_cols; ++j) {
      thread_id = omp_get_thread_num();
      densities[j] = combined_densities(&gpus[thread_id]
                                      , coords
                                      , n_rows
                                      , {j}
                                      , {bandwidths[j]}
                                      , n_wg
                                      , wgsize1d);
    }
    for (Tools::OCL::GPUElement& gpu: gpus) {
      Tools::OCL::cleanup_gpu(&gpu);
    }
#else
    //// compute probability densities with OpenMP on CPU
    for (unsigned int j=0; j < n_selected_cols; ++j) {
      densities[j] = combined_densities(coords
                                      , n_rows
                                      , {j}
                                      , {bandwidths[j]});
    }
#endif
    // normalize densities
    unsigned int i;
    float sum;
    #pragma omp parallel for default(none)\
                             private(j,i,sum)\
                             firstprivate(n_selected_cols,n_rows)\
                             shared(densities)
    for (j=0; j < n_selected_cols; ++j) {
      sum = Tools::kahan_sum(densities[j]);
      for (i=0; i < n_rows; ++i) {
        densities[j][i] /= sum;
      }
    }
    return densities;
  }



  std::tuple<std::vector<std::vector<float>>, std::vector<std::string>>
  compute_densities_2d(std::vector<std::size_t> selected_cols
                     , const float* coords
                     , std::size_t n_rows
                     , std::vector<float> bandwidths) {
    std::vector<std::string> labels;
    std::vector<std::vector<float>> densities;
    unsigned int n_selected_cols = selected_cols.size();
    unsigned int j;
    // compute 2d-probs
    for (std::size_t i=0; i < n_selected_cols; ++i) {
      for (std::size_t j=i+1; j < n_selected_cols; ++j) {
        std::vector<float> dens_ij = combined_densities(coords
                                                      , n_rows
                                                      , {i
                                                        ,j}
                                                      , {bandwidths[i]
                                                       , bandwidths[j]});
        densities.push_back(dens_ij);
        labels.push_back(std::to_string(selected_cols[i])
                       + "_"
                       + std::to_string(selected_cols[j]));
      }
    }
    // normalize densities
    unsigned int i;
    float sum;
    #pragma omp parallel for default(none)\
                             private(j,i,sum)\
                             firstprivate(n_selected_cols,n_rows)\
                             shared(densities)
    for (j=0; j < densities.size(); ++j) {
      sum = Tools::kahan_sum(densities[j]);
      for (i=0; i < n_rows; ++i) {
        densities[j][i] /= sum;
      }
    }
    return std::make_tuple(densities, labels);
  }


  void
  main(boost::program_options::variables_map args) {
    using Tools::IO::selected_coords_bandwidths;
    Tools::IO::set_out(args["output"].as<std::string>());
    std::string fname_input = args["input"].as<std::string>();
    unsigned int dim_kernel = args["dim"].as<unsigned int>();
    if (dim_kernel == 0 || dim_kernel > 3) {
      std::cerr << "error: only allowed kernel dimensions are 1, 2 or 3."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    // read data
    std::vector<std::size_t> selected_cols;
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::vector<float> bandwidths;
    std::tie(selected_cols
           , coords
           , n_rows
           , n_cols
           , bandwidths)
      = selected_coords_bandwidths<float>(fname_input
                                        , args["columns"].as<std::string>()
                                        , args["bandwidths"].as<std::string>());
    // run computation
    std::vector<std::vector<float>> densities;
    if (dim_kernel == 1) {
      densities = compute_densities_1d(selected_cols
                                     , coords
                                     , n_rows
                                     , bandwidths);
      // output: densities
      for (std::size_t i=0; i < n_rows; ++i) {
        for (std::size_t j=0; j < selected_cols.size(); ++j) {
          Tools::IO::out() << " " << densities[j][i];
        }
        Tools::IO::out() << "\n";
      }
    } else if (dim_kernel == 2) {
      std::vector<std::string> labels;
      std::tie(densities, labels) = compute_densities_2d(selected_cols
                                                       , coords
                                                       , n_rows
                                                       , bandwidths);
      for (std::size_t j=0; j < labels.size(); ++j) {
        std::string lbl = labels[j];
        for (std::size_t i=0; i < n_rows; ++i) {
          Tools::IO::out() << lbl << " " << densities[j][i] << "\n";
        }
      }
    } else {
      //TODO 3D
    }
    // cleanup
    Tools::IO::free_coords(coords);
  }

} // end namespace 'Dens'

