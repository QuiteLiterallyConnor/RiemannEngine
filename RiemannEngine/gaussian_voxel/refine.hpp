#ifndef RIEMANN_GAUSSIAN_VOXEL_REFINE_HPP
#define RIEMANN_GAUSSIAN_VOXEL_REFINE_HPP

#include "gaussian_voxel/voxelize.hpp"

#include <string>
#include <vector>

namespace riemann::gaussian_voxel {

enum class RefinementCriterion { Density, OccupancyUncertainty };

struct AdaptiveRefinementOptions {
  RefinementCriterion criterion = RefinementCriterion::OccupancyUncertainty;
  // Density means mass / cell volume. Uncertainty means 4*p*(1-p).
  double score_threshold = 0.25;
  double min_density = 1e-4;
  uint32_t max_depth = 2;
  int64_t max_voxels = 100'000;
  int64_t max_splits_per_iteration = 1024;
  int64_t max_iterations = 2;
};

struct AdaptiveGaussianVoxelOutput {
  GaussianVoxelOutput rasterized;
  std::vector<int64_t> leaf_counts; // input, then each accepted split pass
  std::string stop_reason; // threshold, max_depth, max_voxels, max_iterations
};

// Pilot passes select discrete topology without autograd. The final pass is
// differentiable; neither the reference nor its feature values are mutated.
AdaptiveGaussianVoxelOutput rasterize_adaptive(
    const gaussian::GaussianTensor &gaussians,
    const adaptive::AdaptiveSparseTensor &reference,
    const GaussianVoxelOptions &options = {},
    const AdaptiveRefinementOptions &refinement = {},
    const at::Tensor &confidence = {});

} // namespace riemann::gaussian_voxel

#endif
