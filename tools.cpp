
#include <algorithm>
#include <cctype>

#include "tools.hpp"

namespace Tools {
namespace String {
  std::string
  last(std::size_t n, std::string str) {
    if (str.size() < n) {
      return "";
    } else {
      return str.substr(str.size() - n);
    }
  }
  std::string
  tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
  }
} // end namespace Tools::String

namespace IO {
  void
  set_out(std::string fname) {
    set_stream<std::ofstream&>(fname, ofs);
  }
  std::ostream& out() {
    return get_stream<std::ofstream&, std::ostream&>(ofs, std::cout);
  }
  void
  set_err(std::string fname) {
    set_stream<std::ofstream&>(fname, efs);
  }
  std::ostream& err() {
    return get_stream<std::ofstream&, std::ostream&>(efs, std::cerr);
  }
  void
  set_in(std::string fname) {
    set_stream<std::ifstream&>(fname, ifs);
  }
  std::istream& in() {
    return get_stream<std::ifstream&, std::istream&>(ifs, std::cin);
  }
} // end namespace Tools::IO
} // end namespace Tools

