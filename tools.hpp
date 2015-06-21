#pragma once

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

} // end namespace String
} // end namespace Tools

