#pragma once

#include <iostream>
#include <fstream>
#include <string>

namespace Tools {
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
} // end namespace Tools::String

namespace IO {
  namespace {
    std::ofstream ofs;
    std::ofstream efs;
    std::ifstream ifs;

//TODO docs
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

//TODO docs
  std::ostream&
  out();
  std::ostream&
  err();
  std::istream&
  in();

  void
  set_out(std::string fname);
  void
  set_err(std::string fname);
  void
  set_in(std::string fname);

} // end namespace Tools::IO
} // end namespace Tools

