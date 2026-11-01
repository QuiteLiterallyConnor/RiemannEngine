#ifndef RIEMANN_ENGINE_FOUNDATION_HASHING_HPP
#define RIEMANN_ENGINE_FOUNDATION_HASHING_HPP

#include "engine/foundation/errors.hpp"

#include <cstdint>

namespace riemann {

// Non-owning view of one row of a packed coordinate buffer. Width is carried
// by the hash/equality functors rather than by the view itself.
template <typename coordinate_type> class coordinate {
  using self_type = coordinate<coordinate_type>;

public:
  coordinate() = delete;

  RIEMANN_CUDA_HOST_DEVICE inline coordinate(self_type &other)
      : m_ptr(other.m_ptr) {}
  RIEMANN_CUDA_HOST_DEVICE inline coordinate(self_type const &other)
      : m_ptr(other.m_ptr) {}
  RIEMANN_CUDA_HOST_DEVICE inline coordinate(coordinate_type const *ptr)
      : m_ptr{ptr} {}

  RIEMANN_CUDA_HOST_DEVICE inline coordinate_type const *data() const {
    return m_ptr;
  }
  RIEMANN_CUDA_HOST_DEVICE inline coordinate_type operator[](uint32_t i) const {
    return m_ptr[i];
  }
  RIEMANN_CUDA_HOST_DEVICE inline void data(coordinate_type *ptr) {
    m_ptr = ptr;
  }

private:
  coordinate_type const *m_ptr{nullptr};
};

template <typename T> struct coordinate_print_functor {
  inline coordinate_print_functor(size_t _coordinate_size)
      : m_coordinate_size(_coordinate_size) {}

  std::string operator()(coordinate<T> const &v) {
    Formatter out;
    const char *delim = "";
    out << '[';
    for (size_t i = 0; i < m_coordinate_size; ++i) {
      out << delim << v[i];
      delim = ", ";
    }
    out << "]";
    return out;
  }

  size_t const m_coordinate_size;
};

// Random-access iterator that steps through a packed coordinate buffer one
// row at a time, yielding coordinate views.
template <typename coordinate_type> class coordinate_iterator {
public:
  // clang-format off
  using self_type         = coordinate_iterator<coordinate_type>;
  using size_type         = int32_t;

  using iterator_category = std::random_access_iterator_tag;
  using value_type        = coordinate<coordinate_type>;
  using difference_type   = int32_t;
  using pointer           = coordinate<coordinate_type>*;
  using reference         = coordinate<coordinate_type>&;
  // clang-format on

public:
  coordinate_iterator() = delete;
  RIEMANN_CUDA_HOST_DEVICE coordinate_iterator(coordinate_type const *ptr,
                                               size_type const coordinate_size,
                                               difference_type const steps = 0)
      : m_ptr{ptr}, m_coordinate_size{coordinate_size},
        m_coordinate{m_ptr}, m_steps{steps} {}

  RIEMANN_CUDA_HOST_DEVICE inline reference operator*() noexcept {
    m_coordinate = value_type{m_ptr + m_coordinate_size * m_steps};
    return m_coordinate;
  }
  RIEMANN_CUDA_HOST_DEVICE inline pointer operator->() noexcept {
    m_coordinate = value_type{m_ptr + m_coordinate_size * m_steps};
    return &m_coordinate;
  }

  RIEMANN_CUDA_HOST_DEVICE inline self_type &operator++() noexcept {
    ++m_steps;
    return *this;
  }
  RIEMANN_CUDA_HOST_DEVICE inline self_type operator++(int) noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps + 1};
  }

  RIEMANN_CUDA_HOST_DEVICE inline self_type &operator--() noexcept {
    --m_steps;
    return *this;
  }
  RIEMANN_CUDA_HOST_DEVICE inline self_type operator--(int) noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps - 1};
  }

  RIEMANN_CUDA_HOST_DEVICE inline self_type &
  operator+=(difference_type n) noexcept {
    m_steps += n;
    return *this;
  }
  RIEMANN_CUDA_HOST_DEVICE inline self_type &
  operator-=(difference_type n) noexcept {
    m_steps -= n;
    return *this;
  }

  RIEMANN_CUDA_HOST_DEVICE inline self_type
  operator+(difference_type n) const noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps + n};
  }
  RIEMANN_CUDA_HOST_DEVICE inline self_type
  operator-(difference_type n) const noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps - n};
  }

  RIEMANN_CUDA_HOST_DEVICE inline size_type coordinate_size() const noexcept {
    return m_coordinate_size;
  }

  RIEMANN_CUDA_HOST_DEVICE inline difference_type
  operator-(self_type const &other) const noexcept {
    return m_steps - other.m_steps;
  }

  RIEMANN_CUDA_HOST_DEVICE inline bool
  operator==(self_type const &other) const noexcept {
    return current_position() == other.current_position();
  }

  RIEMANN_CUDA_HOST_DEVICE inline bool
  operator!=(self_type const &other) const noexcept {
    return current_position() != other.current_position();
  }

private:
  inline coordinate_type const *current_position() const noexcept {
    return m_ptr + m_coordinate_size * m_steps;
  }

private:
  coordinate_type const *m_ptr{nullptr};
  size_type const m_coordinate_size = 0;
  value_type m_coordinate;
  difference_type m_steps = 0;
};

template <typename coordinate_type> class coordinate_range {
  using iterator = coordinate_iterator<coordinate_type>;
  using size_type = typename coordinate_iterator<coordinate_type>::size_type;

public:
  coordinate_range() = delete;
  coordinate_range(size_type const number_of_coordinates,
                   size_type const coordinate_size, coordinate_type const *ptr)
      : m_ptr(ptr), m_coordinate_size(coordinate_size),
        m_number_of_coordinates(number_of_coordinates) {}

  iterator begin() const { return iterator(m_ptr, m_coordinate_size); }
  iterator end() const {
    return iterator(m_ptr, m_coordinate_size, m_number_of_coordinates);
  }
  size_type size() { return m_number_of_coordinates; }

private:
  coordinate_type const *m_ptr{nullptr};
  size_type const m_coordinate_size = 0;
  size_type const m_number_of_coordinates = 0;
};

namespace detail {

template <typename coordinate_type> struct coordinate_equal_to {
  RIEMANN_CUDA_HOST_DEVICE inline coordinate_equal_to(size_t _coordinate_size)
      : coordinate_size(_coordinate_size) {}
  RIEMANN_CUDA_HOST_DEVICE inline bool
  operator()(coordinate<coordinate_type> const &lhs,
             coordinate<coordinate_type> const &rhs) const {
    if ((lhs.data() == nullptr) && (rhs.data() == nullptr))
      return true;
    if ((lhs.data() == nullptr) != (rhs.data() == nullptr))
      return false;
    for (size_t i = 0; i < coordinate_size; i++) {
      if (lhs[i] != rhs[i])
        return false;
    }
    return true;
  }

  size_t coordinate_size;
};

// MurmurHash3 x86 32-bit over the raw bytes of one coordinate row. The exact
// bit pattern matters: both host and device tables hash with this functor.
// clang-format off
template <typename coordinate_type> struct coordinate_murmur3 {
  using result_type = uint32_t;

  RIEMANN_CUDA_HOST_DEVICE inline coordinate_murmur3(uint32_t _coordinate_size)
      : m_seed(0), coordinate_size(_coordinate_size),
        len(_coordinate_size * sizeof(coordinate_type)) {}

  RIEMANN_CUDA_HOST_DEVICE inline uint32_t rotl32(uint32_t x, int8_t r) const {
    return (x << r) | (x >> (32 - r));
  }

  RIEMANN_CUDA_HOST_DEVICE inline uint32_t fmix32(uint32_t h) const {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  RIEMANN_CUDA_HOST_DEVICE result_type
  operator()(coordinate<coordinate_type> const &key) const {
    uint8_t const *data   = reinterpret_cast<uint8_t const *>(key.data());
    size_t const nblocks  = len / 4;
    result_type h1        = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;

    // byte-wise reads tolerate unaligned coordinate rows
    auto getblock32 = [] RIEMANN_CUDA_HOST_DEVICE (const uint32_t *p, int i) -> uint32_t {
      auto q = (uint8_t const *)(p + i);
      return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
    };

    uint32_t const *const blocks = (uint32_t const *)(data + nblocks * 4);
    for (size_t i = -nblocks; i; i++) {
      uint32_t k1 = getblock32(blocks, i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }

    uint8_t const *tail = (uint8_t const *)(data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
      [[fallthrough]];
    case 2:
      k1 ^= tail[1] << 8;
      [[fallthrough]];
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
    };

    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
  }

private:
  uint32_t m_seed;
  uint32_t coordinate_size;
  int len;
};
// clang-format on

} // namespace detail

} // namespace riemann

#endif // RIEMANN_ENGINE_FOUNDATION_HASHING_HPP
