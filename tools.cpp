
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

} // end namespace String
} // end namespace Tools

