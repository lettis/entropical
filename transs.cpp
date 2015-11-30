
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <chrono>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <omp.h>

#include "tools.hpp"
#include "density_kernels.hpp"
#include "boxed_search.hpp"

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::variables_map args;
  po::options_description opts(std::string(argv[0]).append(
    " [OPTIONS] FILE\n\n"
    "compute transfer entropies between principal components"));
  try {
    opts.add_options()
      // required options
      ("input,i", po::value<std::string>()->required(), "principal components (required).")
      ("tau,t", po::value<unsigned int>()->default_value(1), "lagtime (in # frames; default: 1).")
      // optional parameters
      ("pcmax", po::value<unsigned int>()->default_value(0), "max. PC to read (default: 0 == read all)")
      ("output,o", po::value<std::string>()->default_value(""), "output file (default: stdout)")
      ("nthreads,n", po::value<unsigned int>()->default_value(0), "number of parallel threads (default: 0 == read from OMP_NUM_THREADS)")
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
    unsigned int nthreads = args["nthreads"].as<unsigned int>();
    if (nthreads > 0) {
      omp_set_num_threads(nthreads);
    }
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
      // read full data set
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C');
    } else {
      // read only until given PC
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<float>(fname_input, 'C', Tools::range<std::size_t>(0, pc_max, 1));
    }
    // compute sigmas (and bandwidths) for every dimension
    std::vector<float> sigmas(n_cols);
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
        // collect resulting sigmas from accumulators
        sigmas[j] = sqrt(variance(acc));
        bandwidths[j] = std::pow(n_rows, -1.0/7.0)*sigmas[j];
      }
    }
    // compute transfer entropies
    std::vector<std::vector<float>> T(n_cols, std::vector<float>(n_cols, 0.0));
    {
      std::size_t x, y, n;
      std::vector<Transs::BoxedSearch::Boxes> searchboxes(n_cols);
      #pragma omp parallel for default(none)\
                               private(x)\
                               firstprivate(n_cols,n_rows)\
                               shared(searchboxes,coords,bandwidths)
      for (x=0; x < n_cols; ++x) {
        searchboxes[x] = Transs::BoxedSearch::Boxes(coords, n_rows, x, bandwidths[x]);
      }
      for (x=0; x < n_cols; ++x) {
        std::cerr << "no of boxes in dim. " << x << ":  " << searchboxes[x].n_boxes() << std::endl;
      }
      enum {X, TAU};
      enum {X_Y, X_XTAU_Y, Y_YTAU_X};
      std::vector<std::array<float, 2>> P(n_cols);
      std::array<float, 3> P_joint;
      #pragma omp parallel for default(none)\
                               private(n,x,y,P_joint)\
                               firstprivate(P,n_cols,n_rows,tau)\
                               shared(std::cout,coords,bandwidths,searchboxes,T)\
                               schedule(dynamic)
      for (n=0; n < n_rows-tau; ++n) {
        // compute p(x_n) and p(x_n, x_n+tau)
        for (x=0; x < n_cols; ++x) {
          using namespace Transs::Epanechnikov;
          std::tie(P[x][X]
                 , P[x][TAU]) = time_lagged_probabilities(n
                                                        , tau
                                                        , coords
                                                        , n_rows
                                                        , x
                                                        , bandwidths[x]
                                                        , searchboxes[x].neighbors_of_state(n));
        }
        // compute joint probabilities
        // p(x_n, y_n),  p(x_n, y_n, x_n+tau),  p(x_n, y_n, y_n+tau)
        for (x=0; x < n_cols; ++x) {
          for (y=x+1; y < n_cols; ++y) {
            using namespace Transs::Epanechnikov;
            std::tie(P_joint[X_Y]
                   , P_joint[X_XTAU_Y]
                   , P_joint[Y_YTAU_X]) = joint_probabilities(n
                                                            , tau
                                                            , coords
                                                            , n_rows
                                                            , x
                                                            , y
                                                            , bandwidths[x]
                                                            , bandwidths[y]
                                                            , Transs::BoxedSearch::joint_neighborhood(searchboxes[x].neighbors_of_state(n)
                                                                                                    , searchboxes[y].neighbors_of_state(n)));
            if (P_joint[X_Y] > 0) {
              if (P_joint[X_XTAU_Y] > 0
               && P[x][TAU] > 0) {
                #pragma omp atomic
                T[y][x] += P_joint[X_XTAU_Y] * log(P_joint[X_XTAU_Y] * P[x][X] / P_joint[X_Y] / P[x][TAU]);
              }
              if (P_joint[Y_YTAU_X] > 0
               && P[y][TAU] > 0) {
                #pragma omp atomic
                T[x][y] += P_joint[Y_YTAU_X] * log(P_joint[Y_YTAU_X] * P[y][X] / P_joint[X_Y] / P[y][TAU]);
              }
            }
            // ... T[x][x] trivially zero
          }
        }
      }
    }
    // normalize by bandwidths
    for (std::size_t x=0; x < n_cols; ++x) {
      for (std::size_t y=0; y < x; ++y) {
        T[y][x] /= n_rows*POW2(bandwidths[x])*bandwidths[y];
        T[x][y] /= n_rows*POW2(bandwidths[y])*bandwidths[x];
      }
    }
    // clean up
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

