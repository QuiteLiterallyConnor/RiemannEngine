#include "gaussian_voxel/refine.hpp"
#include "octree/tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace riemann::gaussian_voxel {

AdaptiveGaussianVoxelOutput rasterize_adaptive(
    const gaussian::GaussianTensor &gaussians,
    const adaptive::AdaptiveSparseTensor &reference,
    const GaussianVoxelOptions &options,
    const AdaptiveRefinementOptions &refinement,
    const at::Tensor &confidence) {
  TORCH_CHECK(reference.dimension() == 3, "adaptive Gaussian voxelization requires 3D");
  TORCH_CHECK(refinement.max_depth <= reference.config().max_depth,
              "refinement max_depth exceeds the reference configuration");
  TORCH_CHECK(refinement.max_voxels >= reference.size() && refinement.max_voxels > 0,
              "max_voxels must be positive and at least the input leaf count");
  TORCH_CHECK(refinement.max_splits_per_iteration > 0 &&
                  refinement.max_iterations >= 0,
              "max_splits_per_iteration must be positive and max_iterations nonnegative");
  TORCH_CHECK(std::isfinite(refinement.score_threshold) &&
                  refinement.score_threshold >= 0 &&
                  std::isfinite(refinement.min_density) && refinement.min_density >= 0,
              "refinement thresholds must be nonnegative and finite");
  TORCH_CHECK(refinement.criterion == RefinementCriterion::Density ||
                  refinement.criterion == RefinementCriterion::OccupancyUncertainty,
              "unknown refinement criterion");
  TORCH_CHECK(refinement.criterion != RefinementCriterion::OccupancyUncertainty ||
                  refinement.score_threshold <= 1,
              "uncertainty score_threshold must be in [0, 1]");

  auto grid = reference;
  std::vector<int64_t> leaf_counts{grid.size()};
  std::string reason = "max_iterations";
  for (int64_t iteration = 0; iteration < refinement.max_iterations; ++iteration) {
    at::NoGradGuard no_grad;
    auto pilot = rasterize(gaussians, grid, options, confidence);
    auto density = pilot.density.squeeze(1) /
                   grid.volumes().to(pilot.density.options());
    auto probability = pilot.occupancy.squeeze(1);
    auto scores = refinement.criterion == RefinementCriterion::Density
                      ? density
                      : 4 * probability * (1 - probability);
    TORCH_CHECK(at::isfinite(scores).all().item<bool>() &&
                    at::isfinite(density).all().item<bool>(),
                "nonfinite refinement scores");
    const auto selected = (density >= refinement.min_density) &
                          (scores > refinement.score_threshold);
    const auto eligible = selected &
        (grid.levels().to(scores.device()) < refinement.max_depth);
    if (!eligible.any().item<bool>()) {
      reason = selected.any().item<bool>() ? "max_depth" : "threshold";
      break;
    }
    // Each octree split replaces one leaf by eight: exact net cost is seven.
    const int64_t capacity = std::min(
        refinement.max_splits_per_iteration,
        (refinement.max_voxels - grid.size()) / 7);
    if (!capacity) {
      reason = "max_voxels";
      break;
    }
    const auto ranked = scores.masked_fill(~eligible, -1);
    // Stable ordering breaks equal-score ties by canonical leaf row.
    const auto order = at::argsort(ranked, true, 0, true);
    const int64_t count = std::min(capacity, eligible.sum().item<int64_t>());
    const auto rows = order.slice(0, 0, count).to(at::kCPU).contiguous();
    std::vector<bool> split(static_cast<size_t>(grid.size()), false);
    for (int64_t i = 0; i < count; ++i)
      split[rows.data_ptr<int64_t>()[i]] = true;

    std::vector<adaptive::detail::Key> keys;
    keys.reserve(static_cast<size_t>(grid.size() + 7 * count));
    for (int64_t row = 0; row < grid.size(); ++row) {
      const auto &key = grid.topology()->keys[row];
      if (split[row]) {
        for (int d = 0; d < 3; ++d)
          TORCH_CHECK(key.coordinates[d] >= std::numeric_limits<int64_t>::min() / 2 &&
                          key.coordinates[d] <= (std::numeric_limits<int64_t>::max() - 1) / 2,
                      "refinement child coordinates would overflow int64");
        adaptive::detail::append_children(key, 3, keys);
      } else {
        keys.push_back(key);
      }
    }
    std::sort(keys.begin(), keys.end(), adaptive::detail::key_before);
    auto topology = adaptive::detail::Topology::make(grid.config(), std::move(keys));
    grid = adaptive::AdaptiveSparseTensor::from_topology(
        topology, at::zeros({topology->size(), 0}, gaussians.positions().options()));
    leaf_counts.push_back(grid.size());
  }
  // max_associations remains a hard error, never a partial/success-shaped result.
  return {rasterize(gaussians, grid, options, confidence), std::move(leaf_counts), reason};
}

} // namespace riemann::gaussian_voxel
