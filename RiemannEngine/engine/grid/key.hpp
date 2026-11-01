#ifndef RIEMANN_ENGINE_GRID_KEY_HPP
#define RIEMANN_ENGINE_GRID_KEY_HPP

#include "engine/foundation/errors.hpp"
#include "engine/foundation/types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace riemann {

/*
 * Handle identifying one coordinate map inside a manager. A key may start
 * unset and be filled in lazily by whichever layer first materializes the
 * map it refers to.
 */
class CoordinateMapKey {
public:
  // clang-format off
  using self_type     = CoordinateMapKey;
  using size_type     = default_types::size_type;
  using stride_type   = default_types::stride_type;
  using hash_key_type = default_types::coordinate_map_hash_type;
  // clang-format on

public:
  CoordinateMapKey() = delete;
  CoordinateMapKey(size_type coordinate_size)
      : m_key_set(false), m_coordinate_size{coordinate_size} {}

  CoordinateMapKey(CoordinateMapKey const &other)
      : m_key_set(other.m_key_set), m_coordinate_size{other.m_coordinate_size},
        m_key(other.m_key) {}

  CoordinateMapKey(size_type coordinate_size,
                   coordinate_map_key_type const &key)
      : m_key_set(true), m_coordinate_size{coordinate_size}, m_key(key) {
    ASSERT(coordinate_size - 1 == m_key.first.size(),
           "Invalid tensor_stride:", m_key.first,
           "coordinate_size:", m_coordinate_size);
  }

  CoordinateMapKey(stride_type tensor_stride, std::string string_id = "")
      : m_coordinate_size(tensor_stride.size() + 1) {
    m_key = std::make_pair(tensor_stride, string_id);
    m_key_set = true;
  }

  size_type get_coordinate_size() const { return m_coordinate_size; }

  void set_key(stride_type tensor_stride, std::string string_id) {
    ASSERT(m_coordinate_size - 1 == tensor_stride.size(),
           "Invalid tensor_stride size:", tensor_stride,
           "coordinate_size:", m_coordinate_size);
    m_key = std::make_pair(tensor_stride, string_id);
    m_key_set = true;
  }

  void set_key(coordinate_map_key_type const &key) {
    ASSERT(m_coordinate_size - 1 == key.first.size(),
           "Invalid tensor_stride size:", key.first,
           "coordinate_size:", m_coordinate_size);
    LOG_DEBUG("Setting the key to ", key.first, ":", key.second);
    m_key = key;
    m_key_set = true;
  }

  coordinate_map_key_type get_key() const {
    ASSERT(is_key_set(), "Key not set");
    return m_key;
  }

  hash_key_type hash() const { return coordinate_map_key_hasher{}(m_key); }

  bool is_key_set() const noexcept { return m_key_set; }

  stride_type get_tensor_stride() const { return m_key.first; }

  bool operator==(CoordinateMapKey const &key) const {
    if (!m_key_set || !key.m_key_set)
      return false;
    return m_key == key.m_key;
  }

  std::string to_string() const {
    Formatter out;
    out << "coordinate map key:" << m_key.first;
    if (m_key.second.length() > 0)
      out << ":" << m_key.second;
    return out;
  }

private:
  bool m_key_set;

  size_type m_coordinate_size;
  coordinate_map_key_type m_key;
}; // CoordinateMapKey

} // namespace riemann

#endif // RIEMANN_ENGINE_GRID_KEY_HPP
