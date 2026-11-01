#ifndef RIEMANN_GAUSSIAN_VOXEL_OCCUPANCY_HPP
#define RIEMANN_GAUSSIAN_VOXEL_OCCUPANCY_HPP

#include "octree/tensor.hpp"

namespace riemann {
namespace occupancy {

struct LidarRayOptions {
  double max_ray_distance = 120.0;
  int64_t max_ray_voxels = 100'000;
};

struct LidarOccupancyTargets {
  at::Tensor occupied; // [B, V] bool
  at::Tensor free;     // [B, V] bool
  at::Tensor observed; // [B, V] bool; unknown is !observed

  at::Tensor labels() const;
};

// Each row is a sensor ray. Points/origins are in world coordinates unless
// sensor_to_world is supplied, in which case both are transformed by the
// matrix indexed by ray_batch_ids. Rays are traversed at the grid's finest
// configured cell size using exact grid-boundary crossings, then assigned to
// the unique active adaptive leaf.
LidarOccupancyTargets from_lidar(
    const at::Tensor &points, const at::Tensor &sensor_origins,
    const at::Tensor &ray_batch_ids, int64_t batch_size,
    const adaptive::AdaptiveSparseTensor &reference,
    const LidarRayOptions &options = {},
    const at::Tensor &sensor_to_world = {});

// Binary cross entropy over observed cells only. Unknown cells are excluded.
// Returns a differentiable zero if the target contains no observed cells.
at::Tensor masked_binary_cross_entropy(
    const at::Tensor &predicted_occupancy,
    const LidarOccupancyTargets &targets, double epsilon = 1e-6);

} // namespace occupancy
} // namespace riemann

#endif
