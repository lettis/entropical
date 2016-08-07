
#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdarg.h>
#include <algorithm>

#include "tools.hpp"

namespace Tools {

  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult) {
    return (unsigned int) std::ceil(orig / ((float) mult));
  };

  std::function<float(float)>
  select_log(bool use_bits) {
    if (use_bits) {
      return [](float x) -> float {
               return std::log2(x);
             };
    } else {
      return [](float x) -> float {
               return std::log(x);
             };
    }
  }

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
      , char delim
      , bool remove_empty) {
    std::istringstream ss(str);
    std::vector<std::string> items;
    std::string item;
    while (std::getline(ss, item, delim)) {
      if ( ! item.empty()) {
        items.push_back(item);
      }
    }
    if (remove_empty) {
      items.erase(std::remove(items.begin(), items.end(), "")
                , items.end());
    }
    return items;
  }

  std::string
  strip(std::string str) {
    std::size_t pos_first_non_space = str.find_first_not_of(" ");
    std::size_t pos_last_non_space = str.find_last_not_of(" ");
    return str.substr(pos_first_non_space, pos_last_non_space - pos_first_non_space);
  }

  //// from: https://github.com/lettis/Kubix
  /**
    behaves like sprintf(char*, ...), but with c++ strings and returns the result
    
    \param str pattern to be printed to
    \return resulting string
    The function internally calls sprintf, but converts the result to a c++ string and returns that one.
    Problems of memory allocation are taken care of automatically.
  */
  std::string
  printf(const std::string& str, ...) {
    unsigned int size = 256;
    va_list args;
    char* buf = (char*) malloc(size * sizeof(char));
    va_start(args, str);
    while (size <= (unsigned int) vsnprintf(buf, size, str.c_str(), args)) {
      size *= 2;
      buf = (char*) realloc(buf, size * sizeof(char));
    }
    va_end(args);
    std::string result(buf);
    free(buf);
    return result;
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

