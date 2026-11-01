#ifndef RIEMANN_OCTREE_TREE_HPP
#define RIEMANN_OCTREE_TREE_HPP

// Internal: the immutable host-side leaf-set description. Every structural
// change builds a new Topology carrying a fresh globally unique id, which is
// also what invalidates cached kernel maps.

#include "cell.hpp"
#include "config.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace riemann {
namespace adaptive {
namespace detail {

constexpr int kMaxDim = static_cast<int>(AdaptiveVoxelConfig::kMaxDimension);
using Key = AdaptiveVoxelKey<kMaxDim>;
using KeyHash = AdaptiveVoxelKeyHash<kMaxDim>;
using DVec = std::array<double, kMaxDim>;

// Canonical leaf order: coarser levels first, then lexicographic coordinates.
inline bool key_before(const Key &a, const Key &b) {
  if (a.level != b.level)
    return a.level < b.level;
  return a.coordinates < b.coordinates;
}

inline Key parent_of(const Key &k) {
  if (k.level == 0)
    return k;
  Key p = k;
  p.level = k.level - 1;
  for (int d = 0; d < kMaxDim; ++d)
    p.coordinates[d] = k.coordinates[d] >> 1;
  return p;
}

// Ancestor of k at `level` (level <= k.level). Arithmetic shifts give floor
// division, matching repeated parent_of.
inline Key coarse_ancestor(const Key &k, uint32_t level) {
  if (level >= k.level)
    return k;
  const uint32_t gap = k.level - level;
  Key a{};
  a.level = level;
  for (int d = 0; d < kMaxDim; ++d)
    a.coordinates[d] = gap >= 63 ? (k.coordinates[d] < 0 ? -1 : 0) : (k.coordinates[d] >> gap);
  return a;
}

inline void append_children(const Key &k, int dim, std::vector<Key> &out) {
  const uint32_t fanout = 1u << dim;
  for (uint32_t i = 0; i < fanout; ++i) {
    Key c{};
    c.level = k.level + 1;
    for (int d = 0; d < dim; ++d)
      c.coordinates[d] = 2 * k.coordinates[d] + ((i >> d) & 1u);
    out.push_back(c);
  }
}

// Stable permutation that sorts `keys` by key_before.
inline std::vector<int64_t> sorted_order(const std::vector<Key> &keys) {
  std::vector<int64_t> order(keys.size());
  for (size_t i = 0; i < order.size(); ++i)
    order[i] = static_cast<int64_t>(i);
  std::stable_sort(order.begin(), order.end(),
                   [&](int64_t a, int64_t b) { return key_before(keys[a], keys[b]); });
  return order;
}

// Runs body(begin, end) over disjoint blocks covering [0, n); large n uses
// several std::thread workers. The first exception escaping body is rethrown.
void parallel_blocks(int64_t n, const std::function<void(int64_t, int64_t)> &body);

// Sorts keys into canonical order. `perm` receives the original position of
// every sorted element and stays empty when the input was already sorted.
void sort_keys(std::vector<Key> &keys, std::vector<int64_t> &perm);

#ifdef USE_CUDA
// Device counterparts (tree_gpu.cu) used for large inputs.
bool cuda_topology_available();
void cuda_sort_keys(std::vector<Key> &keys, std::vector<int64_t> &perm);
bool cuda_has_active_ancestor(const std::vector<Key> &keys,
                              const std::vector<int64_t> &level_offsets, int64_t first);
#endif

struct Topology {
  uint64_t id = 0;
  int D = 0;
  uint32_t max_depth = 0;
  AdaptiveVoxelConfig config; // origin always populated
  DVec origin{};
  DVec base{};
  std::vector<Key> keys; // strictly increasing under key_before
  at::Tensor coords;     // [N, D] int64, CPU
  at::Tensor levels;     // [N] int32, CPU
  at::Tensor volumes;    // [N] double, CPU
  std::vector<int64_t> counts;        // max_depth + 1 entries
  std::vector<int64_t> level_offsets; // max_depth + 2 entries into keys
  std::vector<uint32_t> present_levels;

  int64_t size() const { return static_cast<int64_t>(keys.size()); }

  DVec extent_at(uint32_t level) const {
    const double scale = std::ldexp(1.0, -static_cast<int>(level));
    DVec e{};
    for (int d = 0; d < D; ++d)
      e[d] = base[d] * scale;
    return e;
  }

  void cell_box(const Key &k, DVec &lo, DVec &hi) const {
    const DVec e = extent_at(k.level);
    for (int d = 0; d < D; ++d) {
      lo[d] = origin[d] + static_cast<double>(k.coordinates[d]) * e[d];
      hi[d] = lo[d] + e[d];
    }
  }

  // False when the point maps to a non-finite or out-of-range coordinate.
  bool locate(const DVec &p, uint32_t level, Key &out) const {
    const DVec e = extent_at(level);
    out = Key{};
    out.level = level;
    for (int d = 0; d < D; ++d) {
      const double q = std::floor((p[d] - origin[d]) / e[d]);
      if (!std::isfinite(q) || std::fabs(q) > 4e18)
        return false;
      out.coordinates[d] = static_cast<int64_t>(q);
    }
    return true;
  }

  // Row of the leaf with exactly this key, or -1. Keys are sorted, so a binary
  // search within the level's slice suffices; no index structure is kept.
  int64_t find(const Key &k) const {
    if (k.level > max_depth)
      return -1;
    const Key *base_ptr = keys.data();
    const Key *lo = base_ptr + level_offsets[k.level];
    const Key *hi = base_ptr + level_offsets[k.level + 1];
    const Key *it = std::lower_bound(
        lo, hi, k, [](const Key &a, const Key &b) { return a.coordinates < b.coordinates; });
    if (it != hi && it->coordinates == k.coordinates)
      return it - base_ptr;
    return -1;
  }

  // The unique active leaf owning point p, or -1.
  int64_t find_containing(const DVec &p) const {
    Key probe;
    for (uint32_t l : present_levels) {
      if (!locate(p, l, probe))
        return -1;
      const int64_t row = find(probe);
      if (row >= 0)
        return row;
    }
    return -1;
  }

  // True when some leaf has an active strict ancestor.
  bool has_overlap() const {
    for (const Key &k : keys)
      for (uint32_t l : present_levels) {
        if (l >= k.level)
          break;
        if (find(coarse_ancestor(k, l)) >= 0)
          return true;
      }
    return false;
  }

  // Invokes fn(row, overlap_volume) for every leaf intersecting the half-open
  // box [blo, bhi). Work per present level is proportional to the number of
  // that level's grid cells inside the box.
  template <class Fn> void for_each_overlap(const DVec &blo, const DVec &bhi, Fn &&fn) const {
    for (uint32_t l : present_levels) {
      const DVec e = extent_at(l);
      std::array<int64_t, kMaxDim> first{}, last{};
      bool degenerate = false;
      for (int d = 0; d < D; ++d) {
        const double a = (blo[d] - origin[d]) / e[d];
        const double b = (bhi[d] - origin[d]) / e[d];
        if (!std::isfinite(a) || !std::isfinite(b) || std::fabs(a) > 4e18 || std::fabs(b) > 4e18) {
          degenerate = true;
          break;
        }
        first[d] = static_cast<int64_t>(std::floor(a + 1e-9));
        last[d] = static_cast<int64_t>(std::ceil(b - 1e-9)) - 1;
        if (last[d] < first[d]) {
          degenerate = true;
          break;
        }
      }
      if (degenerate)
        continue;
      std::array<int64_t, kMaxDim> cur = first;
      for (;;) {
        Key probe{};
        probe.level = l;
        probe.coordinates = cur;
        const int64_t row = find(probe);
        if (row >= 0) {
          double part = 1.0;
          for (int d = 0; d < D; ++d) {
            const double cell_lo = origin[d] + static_cast<double>(cur[d]) * e[d];
            const double seg = std::min(bhi[d], cell_lo + e[d]) - std::max(blo[d], cell_lo);
            if (seg <= 1e-9 * e[d]) {
              part = 0.0;
              break;
            }
            part *= seg;
          }
          if (part > 0.0)
            fn(row, part);
        }
        int d = D - 1;
        while (d >= 0 && ++cur[d] > last[d]) {
          cur[d] = first[d];
          --d;
        }
        if (d < 0)
          break;
      }
    }
  }

  // Per-leaf geometry tensors, built on demand (most callers never ask).
  at::Tensor cell_extents() const;                    // [N, D] double
  at::Tensor cell_centers() const;                    // [N, D] double
  std::pair<at::Tensor, at::Tensor> cell_boxes() const; // ([N, D], [N, D]) double

  // `sorted_keys` must be strictly increasing under key_before. Throws
  // std::invalid_argument on duplicates, bad levels, or overlapping leaves.
  static std::shared_ptr<const Topology> make(const AdaptiveVoxelConfig &cfg,
                                              std::vector<Key> sorted_keys);
};

inline bool same_topology(const Topology &a, const Topology &b) {
  if (a.id == b.id)
    return true;
  if (a.D != b.D || a.max_depth != b.max_depth || a.keys.size() != b.keys.size())
    return false;
  return a.origin == b.origin && a.base == b.base && a.keys == b.keys;
}

} // namespace detail
} // namespace adaptive
} // namespace riemann

#endif
