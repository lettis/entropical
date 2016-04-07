
#include <iterator>
#include <map>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/sum_kahan.hpp>

namespace Tools {

  template <typename NUM>
  std::vector<NUM>
  range(NUM from, NUM to, NUM step) {
    std::vector<NUM> result;
    for (NUM i=from; i < to; i += step) {
      result.push_back(i);
    }
    return result;
  }

  template <typename FLOAT>
  FLOAT
  kahan_sum(const std::vector<FLOAT>& xs) {
    using namespace boost::accumulators;
    accumulator_set<FLOAT, features<tag::sum_kahan>> acc;
    for (FLOAT x: xs) {
      acc(x);
    }
    return sum_kahan(acc);
  }

namespace IO {
  template <typename NUM>
  std::tuple<NUM*, std::size_t, std::size_t>
  read_coords(std::string filename
            , char primary_index
            , std::vector<std::size_t> usecols) {
    std::size_t n_rows=0;
    std::size_t n_cols=0;
    std::size_t n_cols_used=0;

    std::ifstream ifs(filename);
    {
      // determine n_cols
      std::string linebuf;
      while (linebuf.empty() && ifs.good()) {
        std::getline(ifs, linebuf);
      }
      std::stringstream ss(linebuf);
      n_cols = std::distance(std::istream_iterator<std::string>(ss),
                             std::istream_iterator<std::string>());
      // go back to beginning to read complete file
      ifs.seekg(0);
      // determine n_rows
      while (ifs.good()) {
        std::getline(ifs, linebuf);
        if ( ! linebuf.empty()) {
          ++n_rows;
        }
      }
      // go back again
      ifs.clear();
      ifs.seekg(0, std::ios::beg);
    }
    std::map<std::size_t, bool> col_used;
    if (usecols.size() == 0) {
      // use all columns
      n_cols_used = n_cols;
      for (std::size_t i=0; i < n_cols; ++i) {
        col_used[i] = true;
      }
    } else {
      // check if given column indices make sense
      for (std::size_t i_col: usecols) {
        if (n_cols <= i_col) {
          std::cerr << "error: given column index "
                    << "is not compatible with available number "
                    << "of columns (" << n_cols << ")"
                    << std::endl;
          exit(EXIT_FAILURE);
        }
      }
      // use only defined columns
      n_cols_used = usecols.size();
      for (std::size_t i=0; i < n_cols; ++i) {
        col_used[i] = false;
      }
      for (std::size_t i: usecols) {
        col_used[i] = true;
      }
    }
    NUM* coords = (NUM*) _mm_malloc(sizeof(NUM)*n_rows*n_cols_used
                                  , MEM_ALIGNMENT);
    ASSUME_ALIGNED(coords);
    // read data
    for (std::size_t cur_row = 0; cur_row < n_rows; ++cur_row) {
      std::size_t cur_col = 0;
      for (std::size_t i=0; i < n_cols; ++i) {
        NUM buf;
        ifs >> buf;
        if (col_used[i]) {
          if (primary_index == 'R') {
            // row-based matrix
            coords[cur_row*n_cols_used + cur_col] = buf;
          } else if (primary_index == 'C') {
            // column-based matrix
            coords[cur_col*n_rows + cur_row] = buf;
          }
          ++cur_col;
        }
      }
    }
    return std::make_tuple(coords, n_rows, n_cols_used);
  }

  template <typename NUM>
  void
  free_coords(NUM* coords) {
    _mm_free(coords);
  }
} // end namespace IO
} // end namespace Tools

