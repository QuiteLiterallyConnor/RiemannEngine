#ifndef RIEMANN_ENGINE_FOUNDATION_SMALL_VECTOR_HPP
#define RIEMANN_ENGINE_FOUNDATION_SMALL_VECTOR_HPP

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>

#include "engine/foundation/errors.hpp"

namespace riemann {

/// Dynamic array that stores up to N elements inline before falling back to
/// heap storage. Only trivially copyable element types are supported.
template <typename T, std::size_t N = 4> class small_vector {
public:
  static_assert(std::is_trivially_copyable<T>::value, "");

  using value_type = T;
  using iterator = T *;
  using const_iterator = T const *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = std::size_t;

  small_vector() {}

  explicit small_vector(size_type sz, T newval = T()) { resize(sz, newval); }

  small_vector(std::initializer_list<T> const init) {
    resize(init.size());
    std::copy(init.begin(), init.end(), data_);
  }

  small_vector(small_vector const &other) {
    resize(other.size_);
    std::copy(other.data_, other.data_ + size_, data_);
  }

  template <typename O> explicit small_vector(small_vector<O> const &other) {
    resize(other.size());
    std::transform(other.data(), other.data() + size_, data_,
                   [](O const &v) { return static_cast<value_type>(v); });
  }

  small_vector(small_vector &&other) noexcept { steal_data_from(other); }

  ~small_vector() { free_array(); }

  small_vector &operator=(small_vector const &other) {
    if (this != &other) {
      resize(other.size_);
      std::copy(other.data_, other.data_ + size_, data_);
    }
    return *this;
  }

  small_vector &operator=(small_vector &&other) noexcept {
    free_array();
    steal_data_from(other);
    return *this;
  }

  void swap(small_vector &other) {
    using std::swap;
    if (is_dynamic()) {
      if (other.is_dynamic()) {
        swap(data_, other.data_);
      } else {
        other.data_ = data_;
        data_ = stat_;
        std::move(other.stat_, other.stat_ + other.size_, stat_);
      }
    } else {
      if (other.is_dynamic()) {
        data_ = other.data_;
        other.data_ = other.stat_;
        std::move(stat_, stat_ + size_, other.stat_);
      } else {
        std::swap_ranges(stat_, stat_ + std::max(size_, other.size_),
                         other.stat_);
      }
    }
    swap(size_, other.size_);
  }

  void resize(size_type newsz, T newval = T()) {
    if (newsz == size_) {
      return;
    }
    if (newsz > static_size_) {
      if (is_dynamic()) {
        T *tmp = static_cast<T *>(std::realloc(data_, newsz * sizeof(T)));
        if (tmp == nullptr) {
          throw std::bad_alloc();
        }
        data_ = tmp;
        if (newsz > size_) {
          std::fill(data_ + size_, data_ + newsz, newval);
        }
        size_ = newsz;
      } else {
        // moving from inline to heap storage; malloc keeps realloc usable
        T *tmp = static_cast<T *>(std::malloc(newsz * sizeof(T)));
        if (tmp == nullptr) {
          throw std::bad_alloc();
        }
        std::move(stat_, stat_ + size_, tmp);
        data_ = tmp;
        std::fill(data_ + size_, data_ + newsz, newval);
        size_ = newsz;
      }
    } else {
      if (is_dynamic()) {
        if (newsz > 0) {
          std::move(data_, data_ + newsz, stat_);
        }
        free_array();
        size_ = newsz;
        data_ = stat_;
      } else {
        if (newsz > size_) {
          std::fill(stat_ + size_, stat_ + newsz, newval);
        }
        size_ = newsz;
      }
    }
  }

  void clear() { resize(0); }

  bool empty() const { return size_ == 0; }

  size_type size() const { return size_; }

  T &operator[](size_type index) { return *(data_ + index); }
  T const &operator[](size_type index) const { return *(data_ + index); }

  T &front() { return *data_; }
  T const &front() const { return *data_; }

  T &back() { return *(data_ + size_ - 1); }
  T const &back() const { return *(data_ + size_ - 1); }

  T *data() { return data_; };
  T const *data() const { return data_; };

  iterator begin() { return data_; }
  const_iterator begin() const { return data_; }
  iterator end() { return data_ + size_; }
  const_iterator end() const { return data_ + size_; }
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  void insert(size_type index, T const &value) {
    ASSERT(index <= size_, "");
    resize(size_ + 1);
    if (index < size_ - 1) {
      std::move_backward(data_ + index, data_ + size_ - 1, data_ + size_);
    }
    *(data_ + index) = value;
  }

  void push_back(T const &value) {
    resize(size_ + 1);
    back() = value;
  }

  void push_back(small_vector const &values) {
    size_type index = size_;
    resize(size_ + values.size_);
    for (size_type ii = 0; ii < values.size_; ++ii) {
      data_[index + ii] = values.data_[ii];
    }
  }

  void erase(size_type index) {
    ASSERT(index < size_, "");
    if (index < size_ - 1) {
      std::move(data_ + index + 1, data_ + size_, data_ + index);
    }
    resize(size_ - 1);
  }

  void pop_back() {
    ASSERT(size_ > 0, "");
    resize(size_ - 1);
  }

private:
  constexpr static size_type static_size_ = N;
  size_type size_ = 0;
  T stat_[static_size_];
  T *data_ = stat_;

  bool is_dynamic() noexcept { return data_ != stat_; }

  void free_array() noexcept {
    if (is_dynamic()) {
      std::free(data_);
    }
  }

  void steal_data_from(small_vector &other) noexcept {
    if (other.is_dynamic()) {
      size_ = other.size_;
      data_ = other.data_;
      other.size_ = 0;
      other.data_ = other.stat_;
    } else {
      size_ = other.size_;
      data_ = stat_;
      std::move(other.data_, other.data_ + size_, data_);
    }
  }
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os,
                                small_vector<T> const &array) {
  os << "{";
  auto it = array.begin();
  if (it != array.end()) {
    os << *it;
    while (++it != array.end()) {
      os << ", " << *it;
    };
  }
  os << "}";
  return os;
}

template <typename T>
inline void swap(small_vector<T> &v1, small_vector<T> &v2) {
  v1.swap(v2);
}

} // namespace riemann

#endif // RIEMANN_ENGINE_FOUNDATION_SMALL_VECTOR_HPP
