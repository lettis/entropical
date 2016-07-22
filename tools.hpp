#pragma once

#define UNUSED(expr) (void)(expr)
#define POW2(X) (X)*(X)

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

// allocate memory (32 bit for SSE4_1, AVX)
// TODO: define MEM_ALIGNMENT during cmake and
//       set depending on usage of SSE2, SSE4_1, AVX or Xeon Phi
#define MEM_ALIGNMENT 32

/// needed for aligned memory allocation for Xeon Phi, SSE or AVX
#if defined(__INTEL_COMPILER)
 #include <malloc.h>
#else
 #include <mm_malloc.h>
#endif

#if defined(__INTEL_COMPILER)
  #define ASSUME_ALIGNED(c) __assume_aligned( (c), MEM_ALIGNMENT)
#else
  #define ASSUME_ALIGNED(c) (c) = (float*) __builtin_assume_aligned( (c), MEM_ALIGNMENT)
#endif

namespace Tools {
  /**
   * emulate python range function
   */
  template <typename NUM>
  std::vector<NUM>
  range(NUM from, NUM to, NUM step);

  /**
   * precise sum of floats by using Kahan summation
   */
  template <typename FLOAT>
  FLOAT
  kahan_sum(const std::vector<FLOAT>& xs);

  /**
   * normalize values to sum
   */
  template <typename FLOAT>
  std::vector<FLOAT>
  sum1_normalized(std::vector<FLOAT> xs);

  /**
   * @returns -1.0 if negative, 1.0 if positive or zero
   */
  template <typename FLOAT>
  FLOAT
  sgn(FLOAT val);

  /**
   * return values of first frame per box
   */
  template <typename NUM>
  std::vector<NUM>
  boxlimits(const std::vector<NUM>& xs
          , std::size_t boxsize
          , std::size_t n_dim);

  template <typename NUM>
  std::pair<std::size_t, std::size_t>
  min_max_box(const std::vector<NUM>& limits
            , NUM val
            , NUM bandwidth);

  /**
   * @returns minimal multiplicator fulfilling a <= min_multiplicator(a,b) * b
   */
  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult);

  /**
   * @returns sorted coordinates of given columns as column-oriented array.
   *
   *          coordinates will be sorted along first column.
   */
  template <typename NUM>
  std::vector<NUM>
  dim1_sorted_coords(const NUM* coords
                   , std::size_t n_rows
                   , std::vector<std::size_t> col_indices);


  namespace String {
    /**
     * return last n characters of str.
     * if str is smaller than n, return an emtpy string.
     */
    std::string
    last(std::size_t n, std::string str);
    /**
     * convert string to all lower case
     */
    std::string
    tolower(std::string s);
    /**
     * split string at given delimiter.
     * if 'remove_empty' is set (default: false),
     * remove all empty strings from result.
     */
    std::vector<std::string>
    split(std::string str
        , char delim
        , bool remove_empty=false);
    /**
     * remove leading and trailing spaces from string
     */
    std::string
    strip(std::string str);
  
    std::string
    printf(const std::string& str, ...);
  } // end namespace Tools::String
  
  
  namespace IO {
    namespace {
      /**
       * filestream object for output stream
       */
      std::ofstream ofs;
      /**
       * filestream object for error stream
       */
      std::ofstream efs;
      /**
       * filestream object for input stream
       */
      std::ifstream ifs;
      /**
       * meta function to open/close file-stream
       */
      template <typename FILE_STREAM>
      void
      set_stream(std::string fname, FILE_STREAM fh) {
        if (fh.is_open()) {
          fh.close();
        }
        if (fname != "") {
          fh.open(fname);
        }
      }
      /**
       * meta function to retrieve file-/console-stream
       * reference depending on stream status.
       */
      template <typename FILE_STREAM, typename STREAM>
      STREAM
      get_stream(FILE_STREAM fh, STREAM console) {
        if (fh.is_open()) {
          return fh;
        } else {
          return console;
        }
      }
    } // end local namespace
  
    /**
     * reference to out-stream.
     * either file (if set) or std::cout.
     */
    std::ostream&
    out();
    /**
     * reference to err-stream.
     * either file (if set) or std::cerr.
     */
    std::ostream&
    err();
    /**
     * reference to in-stream.
     * either file (if set) or std::cin.
     */
    std::istream&
    in();
    /**
     * set out-stream to given file.
     * if fname=="": set to std::cout.
     */
    void
    set_out(std::string fname);
    /**
     * set err-stream to given file.
     * if fname=="": set to std::cerr.
     */
    void
    set_err(std::string fname);
    /**
     * set in-stream to given file.
     * if fname=="": set to std::cin.
     */
    void
    set_in(std::string fname);
    /**
     * read coordinates from space-separated ASCII file.
     * will write data with precision of NUM-type into memory.
     * return value: tuple of
     *   {  data (unique_ptr<NUM> with custom deleter),
     *    n_rows (size_t),
     *    n_cols (size_t)}.
     * usecols defines a preselection of columns (indices based 0).
     */
    template <typename NUM>
    std::tuple<NUM*, std::size_t, std::size_t>
    read_coords(std::string filename,
                char primary_index = 'R',
                std::vector<std::size_t> usecols = {});

    /**
     * read coordinates only of specified columns. 
     * columns are given as space-separated numerical values
     * with 1 as the lowest index.
     * returned coords are generally column oriented.
     */
    template <typename NUM>
    std::tuple<std::vector<std::size_t>, NUM*, std::size_t, std::size_t>
    selected_coords(std::string filename
                  , std::string columns);
    template <typename NUM>
    /**
     * like 'selected_coords', but additionally with bandwidths.
     */
    std::tuple<std::vector<std::size_t>
             , NUM*
             , std::size_t
             , std::size_t
             , std::vector<NUM>>
    selected_coords_bandwidths(std::string fname_coords
                             , std::string columns
                             , std::string fname_bandwidths);
    /**
     * free memory pointing to coordinates
     */
    template <typename NUM>
    void
    free_coords(NUM* coords);
  } // end namespace Tools::IO
} // end namespace Tools

#include "tools.hxx"

