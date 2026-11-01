#ifndef RIEMANN_OCTREE_CELL_HPP
#define RIEMANN_OCTREE_CELL_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace riemann {
namespace adaptive {

// 64-bit finalizer with full avalanche; used to combine key components.
inline uint64_t scramble_bits(uint64_t v) noexcept {
  v ^= v >> 33;
  v *= 0xff51afd7ed558ccdULL;
  v ^= v >> 33;
  v *= 0xc4ceb9fe1a85ec53ULL;
  v ^= v >> 33;
  return v;
}

// Identity of a single cell: integer grid coordinates at its refinement level.
// Keys at different levels are distinct even for equal coordinates.
template <size_t D> struct AdaptiveVoxelKey {
  std::array<int64_t, D> coordinates;
  uint32_t level;

  bool operator==(const AdaptiveVoxelKey &) const = default;
};

template <size_t D> struct AdaptiveVoxelKeyHash {
  size_t operator()(const AdaptiveVoxelKey<D> &k) const noexcept {
    uint64_t acc = scramble_bits(0xa0761d6478bd642fULL ^ static_cast<uint64_t>(k.level));
    for (size_t d = 0; d < D; ++d)
      acc = scramble_bits(acc ^ (static_cast<uint64_t>(k.coordinates[d]) +
                                 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(d + 1)));
    return static_cast<size_t>(acc);
  }
};

// Maps keys to physical space. A cell at `level` has extent
// base_voxel_size / 2^level per axis and occupies the half-open box [lo, hi),
// so every point belongs to exactly one cell of each level.
template <size_t D> struct AdaptiveVoxelGeometry {
  using Key = AdaptiveVoxelKey<D>;
  using Vec = std::array<double, D>;

  Vec origin{};
  Vec base_voxel_size{};
  uint32_t max_depth = 0;

  Vec voxel_size(uint32_t level) const {
    const double scale = std::ldexp(1.0, -static_cast<int>(level));
    Vec out{};
    for (size_t d = 0; d < D; ++d)
      out[d] = base_voxel_size[d] * scale;
    return out;
  }

  std::pair<Vec, Vec> bounds(const Key &k) const {
    const Vec ext = voxel_size(k.level);
    Vec lo{}, hi{};
    for (size_t d = 0; d < D; ++d) {
      lo[d] = origin[d] + static_cast<double>(k.coordinates[d]) * ext[d];
      hi[d] = lo[d] + ext[d];
    }
    return {lo, hi};
  }

  Vec center(const Key &k) const {
    auto [lo, hi] = bounds(k);
    Vec mid{};
    for (size_t d = 0; d < D; ++d)
      mid[d] = 0.5 * (lo[d] + hi[d]);
    return mid;
  }

  // Cell of `level` owning point p; floor division keeps negative coordinates
  // consistent with the half-open convention.
  Key locate(const Vec &p, uint32_t level) const {
    const Vec ext = voxel_size(level);
    Key k{};
    k.level = level;
    for (size_t d = 0; d < D; ++d)
      k.coordinates[d] = static_cast<int64_t>(std::floor((p[d] - origin[d]) / ext[d]));
    return k;
  }

  static Key parent(const Key &k) {
    if (k.level == 0)
      return k;
    Key p{};
    p.level = k.level - 1;
    for (size_t d = 0; d < D; ++d)
      p.coordinates[d] = k.coordinates[d] >> 1; // arithmetic shift: floor halving
    return p;
  }

  static Key child(const Key &k, uint32_t index) {
    Key c{};
    c.level = k.level + 1;
    for (size_t d = 0; d < D; ++d)
      c.coordinates[d] = 2 * k.coordinates[d] + ((index >> d) & 1u);
    return c;
  }

  static std::array<Key, (size_t(1) << D)> children(const Key &k) {
    std::array<Key, (size_t(1) << D)> out{};
    for (uint32_t i = 0; i < out.size(); ++i)
      out[i] = child(k, i);
    return out;
  }

  // Strict ancestry: a sits above b and contains it.
  static bool is_ancestor(const Key &a, const Key &b) {
    if (a.level >= b.level)
      return false;
    const uint32_t gap = b.level - a.level;
    if (gap >= 63)
      return false;
    for (size_t d = 0; d < D; ++d)
      if ((b.coordinates[d] >> gap) != a.coordinates[d])
        return false;
    return true;
  }
};

} // namespace adaptive
} // namespace riemann

#endif
