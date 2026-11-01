#ifndef RIEMANN_ENGINE_FOUNDATION_ERRORS_HPP
#define RIEMANN_ENGINE_FOUNDATION_ERRORS_HPP

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace riemann {

constexpr char const *const ERROR_MAP_NOT_FOUND = "CoordinateMap not found";
constexpr char const *const ERROR_NOT_IMPLEMENTED = "Not implemented";
constexpr char const *const ERROR_CPU_ONLY = "Compiled with only CPU backend.";

struct timer {
  void tic() { m_start = std::chrono::high_resolution_clock::now(); }

  double toc() {
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - m_start)
        .count();
  }

  std::chrono::high_resolution_clock::time_point m_start;
};

template <typename T>
std::ostream &print_vector(std::ostream &out, const T &v) {
  if (!v.empty()) {
    const char *delim = "";
    out << '[';
    for (const auto &elem : v) {
      out << delim << elem;
      delim = ", ";
    }
    out << "]";
  }
  return out;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  return print_vector(out, v);
}

template <typename T> std::string ArrToString(const T &arr) {
  std::string buf = "[";
  bool first = true;
  for (const auto &a : arr) {
    if (!first)
      buf += ", ";
    buf += std::to_string(a);
    first = false;
  }
  buf += "]";
  return buf;
}

template <typename T> std::string PtrToString(const T *ptr, int size) {
  std::string buf = "[";
  for (int i = 0; i < size; i++) {
    buf += (i ? ", " : "") + std::to_string(ptr[i]);
  }
  buf += "]";
  return buf;
}

// Stream-style message builder used by the assertion/exception macros.
class Formatter {
public:
  Formatter() {}
  ~Formatter() {}

  template <typename Type> Formatter &operator<<(const Type &value) {
    stream_ << value;
    return *this;
  }

  template <typename Type> void append(Type value) { stream_ << value; }

  template <typename Type, typename... Args>
  void append(Type value, Args... args) {
    stream_ << value << " ";
    append(args...);
  }

  std::string str() const { return stream_.str(); }
  operator std::string() const { return stream_.str(); }

  enum ConvertToString { to_str };

  std::string operator>>(ConvertToString) { return stream_.str(); }

private:
  std::stringstream stream_;
  Formatter(const Formatter &);
  Formatter &operator=(Formatter &);
};

#define OVERFLOW_IF(condition, ...)                                            \
  {                                                                            \
    if (condition) {                                                           \
      riemann::Formatter formatter;                                            \
      formatter << __FILE__ << ":" << __LINE__ << ",";                         \
      formatter << " overflow condition (" #condition << "). ";                \
      formatter.append(__VA_ARGS__);                                           \
      throw std::overflow_error(formatter.str());                              \
    }                                                                          \
  }

#define ASSERT(condition, ...)                                                 \
  {                                                                            \
    if (!(condition)) {                                                        \
      riemann::Formatter formatter;                                            \
      formatter << __FILE__ << ":" << __LINE__ << ",";                         \
      formatter << " assertion (" #condition << ") failed. ";                  \
      formatter.append(__VA_ARGS__);                                           \
      throw std::runtime_error(formatter.str());                               \
    }                                                                          \
  }

#define WARNING(condition, ...)                                                \
  {                                                                            \
    if (condition) {                                                           \
      riemann::Formatter formatter;                                            \
      formatter << __FILE__ << ":" << __LINE__ << ",";                         \
      formatter << " (" #condition << ") ";                                    \
      formatter.append(__VA_ARGS__);                                           \
      std::cerr << formatter.str() << std::endl;                               \
    }                                                                          \
  }

#ifdef __CUDACC__
#define RIEMANN_CUDA_HOST_DEVICE __host__ __device__
#define RIEMANN_CUDA_DEVICE __device__
#else
#define RIEMANN_CUDA_HOST_DEVICE
#define RIEMANN_CUDA_DEVICE
#ifndef THRUST_CHECK
#define THRUST_CHECK(condition) condition;
#endif
#endif

#define RIEMANN_LOG_COLOR "\033[31;1m"
#define RIEMANN_LOG_RESET "\033[0m"

#ifdef DEBUG
#define __DEBUG(...)                                                           \
  {                                                                            \
    riemann::Formatter formatter;                                              \
    formatter << RIEMANN_LOG_COLOR << __FILE__ << ":" << __LINE__             \
              << RIEMANN_LOG_RESET << " ";                                     \
    formatter.append(__VA_ARGS__);                                             \
    std::cerr << formatter.str() << "\n";                                      \
  }
#define LOG_DEBUG(...) __DEBUG(__VA_ARGS__)
#else
#define LOG_DEBUG(...) (void)0
#endif

#define __WARN(...)                                                            \
  {                                                                            \
    riemann::Formatter formatter;                                              \
    formatter << RIEMANN_LOG_COLOR << "WARNING:" << __FILE__ << ":"           \
              << __LINE__ << RIEMANN_LOG_RESET << " ";                         \
    formatter.append(__VA_ARGS__);                                             \
    std::cerr << formatter.str() << "\n";                                      \
  }

#define LOG_WARN(...) __WARN(__VA_ARGS__)

// Lightweight [from, to) integer range usable in range-based for loops.
class simple_range {
  using index_type = uint32_t;

public:
  class iterator {
    index_type m_num;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = index_type;
    using difference_type = index_type;
    using pointer = const index_type *;
    using reference = index_type;

    explicit iterator(index_type _num = 0) : m_num(_num) {}
    iterator &operator++() {
      m_num++;
      return *this;
    }
    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    int32_t operator-(iterator const &other) const {
      return m_num - other.m_num;
    }
    bool operator==(iterator const &other) const {
      return m_num == other.m_num;
    }
    bool operator!=(iterator const &other) const { return !(*this == other); }
    bool operator!=(iterator &&other) const { return !(*this == other); }
    reference operator*() const { return m_num; }
  };

  simple_range(index_type from, index_type to) : m_from(from), m_to(to) {
    ASSERT(m_to > m_from, "Invalid range");
  }
  simple_range(index_type to) : m_from(0), m_to(to) {
    ASSERT(m_to > m_from, "Invalid range");
  }
  simple_range(simple_range const &) = delete;
  simple_range(simple_range &&other) : m_from(other.m_from), m_to(other.m_to) {}

  std::string to_string() {
    Formatter formatter;
    formatter << "Range: " << m_from << " -- " << m_to << "\n";
    return formatter.str();
  }

  iterator begin() { return iterator(m_from); }
  iterator end() { return iterator(m_to); }

  index_type const m_from, m_to;
};

} // namespace riemann

#endif // RIEMANN_ENGINE_FOUNDATION_ERRORS_HPP
