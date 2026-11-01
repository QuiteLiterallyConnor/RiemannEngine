#ifndef RIEMANN_GAUSSIAN_VOXEL_VOXELIZE_HPP
#define RIEMANN_GAUSSIAN_VOXEL_VOXELIZE_HPP

#include "octree/tensor.hpp"
#include "splat/soup.hpp"

namespace riemann {
namespace gaussian_voxel {

enum class FeatureAggregation {
  WeightedSum,
  NormalizedWeightedMean
};

enum class OccupancyModel {
  Poisson,
  ClampedLinear
};

struct GaussianVoxelOptions {
  double support_radius = 3.0;
  double opacity_density_scale = 1.0;
  // Bounds both emitted Gaussian/voxel pairs and support-index cells examined.
  int64_t max_associations = 10'000'000;
  FeatureAggregation feature_aggregation =
      FeatureAggregation::NormalizedWeightedMean;
  OccupancyModel occupancy_model = OccupancyModel::Poisson;
};

struct GaussianVoxelOutput {
  adaptive::AdaptiveSparseTensor voxels;
  at::Tensor density;   // [V, 1], integrated density mass per voxel
  at::Tensor occupancy; // [V, 1], mapped from density by the selected model
};

// Rasterizes activated Gaussians onto the immutable reference topology.
// The initial reference implementation requires CPU tensors and a 3D grid.
// Support selection is discrete; all values within the selected support are
// differentiable with respect to the Gaussian parameters and optional weights.
GaussianVoxelOutput rasterize(
    const gaussian::GaussianTensor &gaussians,
    const adaptive::AdaptiveSparseTensor &reference,
    const GaussianVoxelOptions &options = {},
    const at::Tensor &confidence = {});

} // namespace gaussian_voxel
} // namespace riemann

#endif
