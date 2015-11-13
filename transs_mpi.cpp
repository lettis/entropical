
#include "transs_mpi.hpp"


int main_mpi(int argc, char* argv[]) {
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
    // initialize MPI
    MPI_Init(NULL, NULL);
    int n_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
    int node_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    // set output stream to file or STDOUT, depending on args
    Tools::IO::set_out(args["output"].as<std::string>()); 
    std::string fname_input = args["input"].as<std::string>();
    unsigned int pc_max = args["pcmax"].as<unsigned int>();
    if (pc_max == 1) {
      if (node_id == MAIN_PROCESS) {
        std::cerr << "error: need at least two PCs to compute information transfer" << std::endl;
      }
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    unsigned int tau = args["tau"].as<unsigned int>();
    if (tau == 0) {
      if (node_id == MAIN_PROCESS) {
        std::cerr << "error: a lagtime of 0 frames does not make sense." << std::endl;
      }
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    // read coordinates
    TRANSS_FLOAT* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::size_t n_cols2 = POW2(n_cols);
    if (pc_max == 0) {
      // read full data set
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<TRANSS_FLOAT>(fname_input, 'C');
    } else {
      // read only until given PC
      std::tie(coords, n_rows, n_cols) = Tools::IO::read_coords<TRANSS_FLOAT>(fname_input, 'C', Tools::range<std::size_t>(0, pc_max, 1));
    }
    // compute sigmas (and bandwidths) for every dimension
    std::vector<TRANSS_FLOAT> sigmas(n_cols);
    std::vector<TRANSS_FLOAT> col_min(n_cols,  std::numeric_limits<TRANSS_FLOAT>::infinity());
    std::vector<TRANSS_FLOAT> col_max(n_cols, -std::numeric_limits<TRANSS_FLOAT>::infinity());
    {
      using namespace boost::accumulators;
      using VarAcc = accumulator_set<TRANSS_FLOAT, features<tag::variance(lazy)>>;
      for (std::size_t j=0; j < n_cols; ++j) {
        VarAcc acc;
        for (std::size_t i=0; i < n_rows; ++i) {
          acc(coords[j*n_rows+i]);
          col_min[j] = std::min(col_min[j], coords[j*n_rows+i]);
          col_max[j] = std::max(col_max[j], coords[j*n_rows+i]);
        }
        // collect resulting sigmas from accumulators
        sigmas[j] = sqrt(variance(acc));
      }
    }
    // compute transfer entropies
    std::vector<TRANSS_FLOAT> T(n_cols2, 0.0);
    {
      std::size_t y, x, n, i;
      TRANSS_FLOAT p_s_y, p_s_x;
      TRANSS_FLOAT s_xxy, s_xy, s_xx, s_x;
      TRANSS_FLOAT tmp_xn, tmp_xn_tau, tmp_yn;
      for (y=0; y < n_cols; ++y) {
        for (x=0; x < n_cols; ++x) {
          // distribute work in equal shares to all MPI nodes
          if (((y*n_cols+x) % n_nodes) == node_id) {
            //  partial prefactors
            p_s_y = -1.0 / (2*std::pow(n_rows, -2.0/7.0)*POW2(sigmas[y]));
            p_s_x = -1.0 / (2*std::pow(n_rows, -2.0/7.0)*POW2(sigmas[x]));
            // compute local sums for every frame n
            #pragma omp parallel for default(none)\
                                     private(n,i,s_xxy,s_xy,s_xx,s_x,tmp_xn,tmp_xn_tau,tmp_yn)\
                                     firstprivate(x,y,n_rows,tau)\
                                     shared(coords,sigmas,T_part)
            for (n=0; n < n_rows-tau; ++n) {
              // partial sums
              s_xxy = 0.0;
              s_xy = 0.0;
              s_xx = 0.0;
              s_x = 0.0;
              // compute partial sums with fixed reference frame n
              #pragma simd reduction(+:s_xxy,s_xy,s_xx,s_x)
              for (i=0; i < n_rows; ++i) {
                tmp_xn = exp(p_s_x * POW2(coords[x*n_rows+n] - coords[x*n_rows+i]));
                tmp_xn_tau = exp(p_s_x * POW2(coords[x*n_rows+n+tau] - coords[x*n_rows+i]));
                tmp_yn = exp(p_s_y * POW2(coords[y*n_rows+n] - coords[y*n_rows+i]));
                s_xxy += tmp_xn_tau * tmp_xn * tmp_yn;
                s_xy += tmp_xn * tmp_yn;
                s_xx += tmp_xn * tmp_xn;
                s_x += tmp_xn;
              }
              T_part[n] = s_xxy * log(2*M_PI * s_xxy * s_x / s_xy / s_xx);
            }
            #pragma omp parallel for default(none)\
                                     private(n)\
                                     firstprivate(x,y)\
                                     shared(T,T_part)\
                                     reduction(+:T)
            for (n=0; n < n_rows; ++n) {
              T[y*n_cols+x] += T_part[n]
            }
            T[y*n_cols+x] *= std::pow(n_rows, -4.0/7.0) / (std::pow(2*M_PI, 3.0/2.0)*sigmas[x]*sigmas[x]*sigmas[y]);
          }
        }
      }
    }
    // accumulate T in main MPI process
    MPI_Barrier(MPI_COMM_WORLD);
    if (node_id == MAIN_PROCESS) {
      for (int i_slave=1; i_slave < n_nodes; ++i_slave) {
        std::vector<TRANSS_FLOAT> T_buf(n_cols2);
        //MPI_Recv(T_buf.data(), n_cols2, MPI_DOUBLE, i_slave, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(T_buf.data(), n_cols2, MPI_FLOAT, i_slave, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        #pragma omp parallel for default(none)\
                                 private(i)\
                                 firstprivate(n_cols2)\
                                 shared(T,T_buf)
        for (i=0; i < n_cols2; ++i) {
          T[i] += T_buf[i];
        }
      }
    } else {
      //MPI_Send(T.data(), n_cols2, MPI_DOUBLE, MAIN_PROCESS, 0, MPI_COMM_WORLD);
      MPI_Send(T.data(), n_cols2, MPI_FLOAT, MAIN_PROCESS, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // clean up
    Tools::IO::free_coords(coords);
    // output
    if (node_id == MAIN_PROCESS) {
      for (std::size_t y=0; y < n_cols; ++y) {
        for (std::size_t x=0; x < n_cols; ++x) {
          Tools::IO::out() << " " << T[y*n_cols+x];
        }
        Tools::IO::out() << "\n";
      }
    }
  } catch (const boost::bad_any_cast& e) {
    if (node_id == MAIN_PROCESS) {
      if ( ! args["help"].as<bool>()) {
        std::cerr << "\nerror parsing arguments!\n\n";
      }
      std::cerr << opts << std::endl;
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    if (node_id == MAIN_PROCESS) {
      std::cerr << "\n" << e.what() << "\n\n";
      std::cerr << opts << std::endl;
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}

