
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <omp.h>

#include "transs.hpp"
#include "tools.hpp"
#include "boxedsearch.hpp"
#include "transs_opencl.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS] FILE\n\n"
    "compute transfer entropies between principal components (or other observables)"));
  try {
    opts.add_options()
      // required options
      ("input,i", po::value<std::string>()->required(), "principal components (required).")
      ("tau,t", po::value<unsigned int>()->default_value(1), "lagtime (in # frames; default: 1).")
      // optional parameters
      ("pcmax", po::value<unsigned int>()->default_value(0), "max. PC to read (default: 0 == read all)")
      ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
      ("wgsize", po::value<unsigned int>()->default_value(64), "workgroup size (default: 64)")
      ("verbose,v", po::bool_switch()->default_value(false), "verbose output.")
      ("help,h", po::bool_switch()->default_value(false), "show this help.");
    // option parsing, settings, checks
    po::positional_options_description pos_opts;
    pos_opts.add("input", 1);
    po::store(po::command_line_parser(argc, argv).options(opts).positional(pos_opts).run(), args);
    po::notify(args);
    if (args["help"].as<bool>()) {
      std::cout << opts << std::endl;
      return EXIT_SUCCESS;
    }
    bool verbose = args["verbose"].as<bool>();
    unsigned int wgsize = args["wgsize"].as<unsigned int>();
    // set output stream to file or STDOUT, depending on args
    Tools::IO::set_out(args["output"].as<std::string>()); 
    std::string fname_input = args["input"].as<std::string>();
    unsigned int pc_max = args["pcmax"].as<unsigned int>();
    if (pc_max == 1) {
      std::cerr << "error: need at least two PCs to compute information transfer" << std::endl;
      return EXIT_FAILURE;
    }
    unsigned int tau = args["tau"].as<unsigned int>();
    if (tau == 0) {
      std::cerr << "error: a lagtime of 0 frames does not make sense." << std::endl;
      return EXIT_FAILURE;
    }
    // read coordinates
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    if (pc_max == 0) {
      verbose && std::cout << "reading full dataset" << std::endl;
      // read full data set
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C');
    } else {
      // read only until given PC
      verbose && std::cout << "reading dataset up to PC " << pc_max << std::endl;
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C', Tools::range<std::size_t>(0, pc_max, 1));
    }
    // compute bandwidths for every dimension
    verbose && std::cout << "computing bandwidths" << std::endl;
    std::vector<float> bandwidths(n_cols);
    std::vector<float> col_min(n_cols,  std::numeric_limits<float>::infinity());
    std::vector<float> col_max(n_cols, -std::numeric_limits<float>::infinity());
    {
      using namespace boost::accumulators;
      using VarAcc = accumulator_set<float, features<tag::variance(lazy)>>;
      for (std::size_t j=0; j < n_cols; ++j) {
        VarAcc acc;
        for (std::size_t i=0; i < n_rows; ++i) {
          acc(coords[j*n_rows+i]);
          col_min[j] = std::min(col_min[j], coords[j*n_rows+i]);
          col_max[j] = std::max(col_max[j], coords[j*n_rows+i]);
        }
        bandwidths[j] = std::pow(n_rows, -1.0/7.0)*sqrt(variance(acc));
      }
    }
    // compute transfer entropies
    std::vector<std::vector<float>> T(n_cols, std::vector<float>(n_cols, 0.0));
    {
      // compute search boxes for fast neighbor search
      std::size_t x, y, thread_id;

/*
TODO: check if box-assisted search helps with kernel performance
      std::vector<Transs::BoxedSearch::Boxes> searchboxes(n_cols);
      #pragma omp parallel for default(none)\
                               private(x)\
                               firstprivate(n_cols,n_rows)\
                               shared(searchboxes,coords,bandwidths)
      for (x=0; x < n_cols; ++x) {
        searchboxes[x] = Transs::BoxedSearch::Boxes(coords, n_rows, x, bandwidths[x]);
      }
      for (x=0; verbose && (x < n_cols); ++x) {
        std::cerr << "no of boxes in dim. " << x << ":  " << searchboxes[x].n_boxes() << std::endl;
      }
*/

      // OpenCL setup
      unsigned int n_workgroups = (unsigned int) std::ceil(n_rows / ((float) wgsize));
      verbose && std::cout << "(n_workgroups, n, n_extended): "
                           << "(" << n_workgroups
                           << ", " << n_rows
                           << ", " << n_workgroups*wgsize << ")" << std::endl;
      std::vector<Transs::OCL::GPUElement> gpus = Transs::OCL::gpus();
      std::size_t n_gpus = gpus.size();
      if (n_gpus == 0) {
        std::cerr << "error: no GPUs found for OpenCL transfer entropy computation" << std::endl;
        return EXIT_FAILURE;
      } else {
        verbose && std::cout << "computing transfer entropies on " << n_gpus << " GPUs" << std::endl;
      }
      std::string kernel_src = Transs::OCL::load_kernel_source("transs.cl");
      for(Transs::OCL::GPUElement& gpu: gpus) {
        Transs::OCL::setup_gpu(gpu, kernel_src, wgsize, n_workgroups, n_rows);
      }

      #pragma omp parallel for default(none)\
                               private(x,y,thread_id)\
                               firstprivate(tau,n_rows,n_cols,n_workgroups,wgsize)\
                               shared(coords,bandwidths,gpus,T)\
                               num_threads(n_gpus)\
                               collapse(2)\
                               schedule(dynamic, 1)
      for (x=0; x < n_cols; ++x) {
        for (y=0; y < n_cols; ++y) {
          if (x < y) {
            thread_id = omp_get_thread_num();
            std::pair<float, float> _T = Transs::OCL::transfer_entropies(gpus[thread_id]
                                                                       , x
                                                                       , y
                                                                       , coords
                                                                       , n_rows
                                                                       , tau
                                                                       , bandwidths
                                                                       , wgsize
                                                                       , n_workgroups);
            T[x][y] = _T.first;
            T[y][x] = _T.second;
            // T[x][x] == 0  by construction
          }
        }
      }
      // OpenCL cleanup
      for (Transs::OCL::GPUElement& gpu: gpus) {
        Transs::OCL::cleanup_gpu(gpu);
      }
    }
    // memory cleanup
    Tools::IO::free_coords(coords);
    // output
    for (std::size_t y=0; y < n_cols; ++y) {
      for (std::size_t x=0; x < n_cols; ++x) {
        Tools::IO::out() << " " << T[y][x];
      }
      Tools::IO::out() << "\n";
    }
  } catch (const boost::bad_any_cast& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments!\n\n";
    }
    std::cerr << opts << std::endl;
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "\n" << e.what() << "\n\n";
    std::cerr << opts << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

