#ifndef _ERROR_
#define _ERROR_
#include <stdexcept>
#include <stream>
#include <string>
class ocl_error : public std::runtime_error {
public:
  ocl_error(const std::string &msg) : std::runtime_error(msg) {}
};
class parser_error : public std::runtime_error {
public:
  parser_error(const std::string &msg) : std::runtime_error(msg) {}
};

template <typename T, typename... Args>
std::string make_msg(const std::string &msg, T val, Args... args) {
  std::stringstream ss;
  std::string format_str(msg);
  ss << " " << val;
  format_str += ss.str();
  format_str = make_msg(format_str, args...);
  return format_str;
}
#endif // _ERROR_