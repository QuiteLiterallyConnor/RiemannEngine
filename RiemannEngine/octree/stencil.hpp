#ifndef RIEMANN_OCTREE_STENCIL_HPP
#define RIEMANN_OCTREE_STENCIL_HPP

#include "tensor.hpp"

#include <array>
#include <memory>
#include <vector>

namespace riemann {
namespace adaptive {

enum class AdaptiveConvMode {
  // Sample the leaf containing each tap point, weight 1. The default; equals
  // the ordinary sparse convolution on uniform aligned grids.
  CenterSampled,
  // Same neighbors, weighted V_in / V_out.
  VolumeWeighted,
  // Every leaf overlapping the tap box, weighted overlap volume / V_out.
  OverlapWeighted,
};

struct AdaptiveKernelConfig {
  int kernel_size = 3; // odd, per dimension
  AdaptiveConvMode mode = AdaptiveConvMode::CenterSampled;
  // 0 keeps input leaves as outputs; -n coarsens outputs n levels (never below
  // level 0); +n subdivides every leaf n levels. Refinement must fit max_depth
  // and int64 coordinates; output count grows by 2^(dimension * n).
  int output_level_shift = 0;
};

// Sparse (out, in, kernel slot, weight) tuples grouped by kernel slot; within a
// slot, outputs appear in ascending order.
struct AdaptiveKernelMap {
  at::Tensor in_index;     // [P] int64
  at::Tensor out_index;    // [P] int64
  at::Tensor kernel_index; // [P] int64
  at::Tensor weight;       // [P] double
  std::vector<int64_t> kernel_offsets; // num_kernels + 1 prefix offsets
  int64_t num_kernels = 0;
};

struct KernelMapEntry {
  AdaptiveKernelMap map;
  std::shared_ptr<const detail::Topology> out_topology;
};

// Slot of an offset in {-r..r}^D with r = (kernel_size-1)/2, first dimension slowest.
int64_t kernel_index(std::array<int, 3> offset, int kernel_size);

AdaptiveKernelMap build_adaptive_kernel_map(const AdaptiveSparseTensor &input,
                                            const AdaptiveKernelConfig &config);

// Keyed by (input topology id, kernel size, mode, output shift); topology
// changes mint new ids, so stale entries are never served.
class KernelMapCache {
public:
  struct Stats {
    size_t hits = 0;
    size_t misses = 0;
    size_t entries = 0;
  };

  static KernelMapCache &instance();
  ~KernelMapCache();

  KernelMapEntry get_or_build(const AdaptiveSparseTensor &input,
                              const AdaptiveKernelConfig &config);
  void clear();
  Stats stats() const;
  void set_capacity(size_t max_entries);

private:
  KernelMapCache();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace adaptive
} // namespace riemann

#endif
