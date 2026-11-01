#ifndef RIEMANN_OCTREE_TENSOR_HPP
#define RIEMANN_OCTREE_TENSOR_HPP

#include "cell.hpp"
#include "config.hpp"

#include <torch/torch.h>

#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace riemann {
namespace adaptive {

namespace detail {
struct Topology;
}

// The points (and per-point features) a tensor was voxelized from. Required by
// the point-driven refinement criteria and revoxelize_from_points().
struct SourcePoints {
  at::Tensor points;   // [P, D] double, CPU
  at::Tensor features; // [P, C]
};

// Sparse tensor whose active cells (leaves) may sit at different refinement
// levels; leaves never overlap. The topology is immutable and shared between
// copies: refine/coarsen swap in a new one, so copies stay independent.
class AdaptiveSparseTensor {
public:
  static AdaptiveSparseTensor from_points(const at::Tensor &points,
                                          const at::Tensor &features,
                                          const AdaptiveVoxelConfig &config);
  static AdaptiveSparseTensor from_voxels(const at::Tensor &coordinates,
                                          const at::Tensor &levels,
                                          const at::Tensor &features,
                                          const AdaptiveVoxelConfig &config);
  static AdaptiveSparseTensor
  from_topology(std::shared_ptr<const detail::Topology> topology, at::Tensor features,
                std::shared_ptr<const SourcePoints> source = nullptr);

  int64_t size() const;
  int dimension() const;
  int64_t channels() const;
  const AdaptiveVoxelConfig &config() const;
  torch::Device device() const { return features_.device(); }

  // Leaves in deterministic (level, coordinates) order. Coordinate, level and
  // geometry tensors are CPU-resident and carry no gradients.
  at::Tensor coordinates() const; // [N, D] int64
  at::Tensor levels() const;      // [N] int32
  at::Tensor voxel_sizes() const; // [N, D] double
  at::Tensor volumes() const;     // [N] double
  at::Tensor centers() const;     // [N, D] double
  std::pair<at::Tensor, at::Tensor> bounds() const; // (lo, hi), half-open
  const at::Tensor &features() const { return features_; }
  void set_features(at::Tensor features);

  std::vector<int64_t> counts_by_level() const; // max_depth + 1 entries
  // Row of the active leaf with exactly this key, or -1.
  int64_t index_of(std::array<int64_t, 3> coordinates, uint32_t level) const;
  int64_t index_of_key(const std::vector<int64_t> &coordinates, uint32_t level) const;
  // Changes whenever the leaf set changes; no-op operations leave it alone.
  uint64_t topology_version() const;
  bool has_overlapping_leaves() const;
  const std::shared_ptr<const detail::Topology> &topology() const { return topology_; }
  bool has_source_points() const { return source_ != nullptr; }

  // Replaces leaves intersecting `region` by all 2^D children until they reach
  // target_level; already-fine leaves stay. Replicate copies the parent value
  // to each child, Split divides it by 2^D (preserving sums).
  void refine(const AxisAlignedRegion &region, uint32_t target_level,
              RefinePolicy policy = RefinePolicy::Replicate);
  // Bottom-up merge towards target_level of every parent whose box lies wholly
  // inside `region`; only occupied descendants feed the reduction.
  void coarsen(const AxisAlignedRegion &region, uint32_t target_level,
               Reduction reduction = Reduction::Mean);
  // Need source points (from_points). Refines one level at a time, capped at
  // max_depth, while the criterion holds; children replicate the parent.
  void refine_by_density(int64_t points_per_leaf_threshold);
  void refine_by_variance(double feature_variance_threshold);
  void refine_if(const std::function<bool(int64_t leaf, const std::array<double, 4> &lo,
                                          const std::array<double, 4> &hi, uint32_t level)>
                     &criterion);
  // Recomputes the current leaves' features from the source points (per-leaf
  // mean; leaves containing no point become zero).
  void revoxelize_from_points();

  AdaptiveSparseTensor with_features(at::Tensor features) const;
  AdaptiveSparseTensor to(torch::Device device) const;
  AdaptiveSparseTensor resample_uniform(uint32_t level,
                                        Reduction reduction = Reduction::Mean) const;
  // Re-expresses this tensor on the reference leaves: finer sources reduce,
  // coarser ones replicate, uncovered reference leaves become zero.
  AdaptiveSparseTensor resample_to(const AdaptiveSparseTensor &reference,
                                   Reduction reduction = Reduction::Mean) const;

  // Element-wise; both operands must share the topology (use resample_to).
  AdaptiveSparseTensor operator+(const AdaptiveSparseTensor &other) const;

private:
  AdaptiveSparseTensor() = default;

  std::shared_ptr<const detail::Topology> topology_;
  at::Tensor features_;
  std::shared_ptr<const SourcePoints> source_;
};

AdaptiveSparseTensor concatenate(const AdaptiveSparseTensor &a,
                                 const AdaptiveSparseTensor &b);

} // namespace adaptive
} // namespace riemann

#endif
