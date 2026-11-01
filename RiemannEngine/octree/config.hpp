#ifndef RIEMANN_OCTREE_CONFIG_HPP
#define RIEMANN_OCTREE_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace riemann {
namespace adaptive {

enum class Reduction { Mean, Sum, Max };
enum class RefinePolicy { Replicate, Split };

// Axis-aligned half-open box [lo, hi).
struct AxisAlignedRegion {
  std::vector<double> lo;
  std::vector<double> hi;

  static AxisAlignedRegion everything(size_t dimension);
};

struct AdaptiveVoxelConfig {
  // Spatial dimensions 1..kMaxDimension are supported.
  static constexpr size_t kMaxDimension = 4;
  // Children double coordinates per level, so depth stays far from the point
  // where level-0 coordinates could overflow int64 after 2^depth scaling.
  static constexpr uint32_t kMaxSupportedDepth = 30;

  std::vector<double> base_voxel_size;
  // Optional. When present, base_voxel_size / min_voxel_size must equal
  // 2^max_depth per axis.
  std::vector<double> min_voxel_size;
  // Optional; empty means the zero vector.
  std::vector<double> origin;
  uint32_t max_depth = 0;

  size_t dimension() const { return base_voxel_size.size(); }

  // Throws std::invalid_argument naming the first problem found.
  void validate() const;
};

} // namespace adaptive
} // namespace riemann

#endif
