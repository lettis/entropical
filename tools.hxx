
#include <iterator>
#include <map>
#include <algorithm>

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

  template <typename FLOAT>
  FLOAT
  sgn(FLOAT val) {
    if (val < 0.0) {
      return -1.0;
    } else {
      return  1.0;
    }
  }

  template <typename NUM>
  std::vector<NUM>
  boxlimits(const std::vector<NUM>& xs
          , std::size_t boxsize
          , std::size_t n_dim) {
    std::size_t n_xs = xs.size() / n_dim;
    std::size_t n_boxes = n_xs / boxsize;
    if (n_boxes * boxsize < n_xs) {
      ++n_boxes;
    }
    std::vector<NUM> boxlimits(n_boxes);
    for (std::size_t i=0; i < n_boxes; ++i) {
      // split into boxes on 1st dimension
      // (i.e. col-index == 0)
      boxlimits[i] = xs[i*boxsize];
    }
    return boxlimits;
  }

  template <typename NUM>
  std::pair<std::size_t, std::size_t>
  min_max_box(const std::vector<NUM>& limits
            , NUM val
            , NUM bandwidth) {
    std::size_t n_boxes = limits.size();
    if (n_boxes == 0) {
      return {0,0};
    } else {
      std::size_t i_min = n_boxes - 1;
      std::size_t i_max = 0;
      NUM lbound = val - bandwidth;
      NUM ubound = val + bandwidth;
      for (std::size_t i=1; i < n_boxes; ++i) {
        if (lbound < limits[i]) {
          i_min = i-1;
          break;
        }
      }
      for (std::size_t i=n_boxes; 0 < i; --i) {
        if (limits[i-1] < ubound) {
          i_max = i-1;
          break;
        }
      }
      return {i_min, i_max};
    }
  }

  template <typename NUM>
  std::vector<NUM>
  dim1_sorted_coords(const NUM* coords
                   , std::size_t n_rows
                   , std::vector<std::size_t> col_indices) {
    std::size_t n_dim = col_indices.size();
    std::vector<NUM> sorted_coords(n_rows*n_dim);
    if (n_dim == 1) {
      // directly sort on data if just one column
      std::size_t i_col = col_indices[0];
      for (std::size_t i=0; i < n_rows; ++i) {
        sorted_coords[i] = coords[i_col*n_rows + i];
      }
      std::sort(sorted_coords.begin(), sorted_coords.end());
    } else {
      // sort on first index
      std::vector<std::vector<NUM>> c_tmp(n_rows
                                        , std::vector<float>(n_dim));
      std::sort(c_tmp.begin()
              , c_tmp.end()
              , [] (const std::vector<NUM>& lhs
                  , const std::vector<NUM>& rhs) {
                  return lhs[0] < rhs[0];
                });
      // feed sorted data into 1D-array
      for (std::size_t i=0; i < n_rows; ++i) {
        for (std::size_t j=0; j < n_dim; ++j) {
          sorted_coords[j*n_rows+i] = c_tmp[i][j];
        }
      }
    }
    return sorted_coords;
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
  std::tuple<std::vector<std::size_t>, NUM*, std::size_t, std::size_t>
  selected_coords(std::string filename
                , std::string columns) {
    std::vector<std::size_t> selected_cols = {};
    if (columns.size() > 0) {
      for (std::string c: Tools::String::split(columns, ' ', true)) {
        selected_cols.push_back(std::stoul(c));
      }
    }
    std::vector<std::size_t> selected_cols_base0;
    for (std::size_t c: selected_cols) {
      if (c <= 0) {
        std::cerr << "error: specify column in range of [1, N_COLUMNS]!"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      selected_cols_base0.push_back(c-1);
    }
    NUM* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    std::tie(coords, n_rows, n_cols) =
      Tools::IO::read_coords<NUM>(filename, 'C', selected_cols_base0);
    if (selected_cols.size() == 0) {
      selected_cols = Tools::range<std::size_t>(1, n_cols+1, 1);
    }
    return std::make_tuple(selected_cols, coords, n_rows, n_cols);
  }

  template <typename NUM>
  void
  free_coords(NUM* coords) {
    _mm_free(coords);
  }
} // end namespace IO
} // end namespace Tools

