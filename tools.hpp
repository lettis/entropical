#pragma once

#define POW2(X) (X)*(X)

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

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
     * split string at given delimiter
     */
    std::vector<std::string>
    split(std::string str
        , char delim);
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
     * return value: tuple of {data (unique_ptr<NUM> with custom deleter),
     * n_rows (size_t), n_cols (size_t)}.
     */
    template <typename NUM>
    std::tuple<NUM*, std::size_t, std::size_t>
    read_coords(std::string filename,
                char primary_index = 'R',
                std::vector<std::size_t> usecols = {});
    /**
     * free memory pointing to coordinates
     */
    template <typename NUM>
    void
    free_coords(NUM* coords);
  } // end namespace Tools::IO
} // end namespace Tools

#include "tools.hxx"

