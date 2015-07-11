
#include <algorithm>
#include <cctype>
#include <sstream>

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

  std::vector<std::string>
  split(std::string str
      , char delim) {
    std::istringstream ss(str);
    std::vector<std::string> items;
    std::string item;
    while (std::getline(ss, item, delim)) {
      if ( ! item.empty()) {
        items.push_back(item);
      }
    }
    return items;
  }

  std::string
  strip(std::string str) {
    std::size_t pos_first_non_space = str.find_first_not_of(" ");
    std::size_t pos_last_non_space = str.find_last_not_of(" ");
    return str.substr(pos_first_non_space, pos_last_non_space - pos_first_non_space);
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

