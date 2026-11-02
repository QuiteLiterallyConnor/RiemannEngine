#include "gaussian_voxel/occupancy.hpp"

#include "octree/tree.hpp"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace riemann {
namespace occupancy {
namespace {

using Vec3 = std::array<double, 3>;

Vec3 read_vec(const double *values, int64_t row) {
  return {values[row * 3], values[row * 3 + 1], values[row * 3 + 2]};
}

Vec3 transform_point(const Vec3 &point, const double *matrix, int64_t row) {
  Vec3 result{};
  const double *m = matrix + row * 16;
  for (int r = 0; r < 3; ++r)
    result[r] = m[r * 4] * point[0] + m[r * 4 + 1] * point[1] +
                m[r * 4 + 2] * point[2] + m[r * 4 + 3];
  return result;
}

void require_finite_points(const at::Tensor &points, const char *name) {
  TORCH_CHECK(points.device().is_cpu(),
              name, " must be a CPU tensor for LiDAR target generation");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
              name, " must have shape [R, 3]");
  TORCH_CHECK(points.is_floating_point() &&
                  (points.scalar_type() == at::kFloat ||
                   points.scalar_type() == at::kDouble),
              name, " must use float32 or float64");
  TORCH_CHECK(at::isfinite(points).all().item<bool>(),
              name, " must contain only finite values");
}

void mark_ray(const adaptive::detail::Topology &topology,
              const Vec3 &origin, const Vec3 &endpoint,
              double max_distance, int64_t max_voxels,
              std::vector<uint8_t> &free, int64_t batch,
              std::vector<uint8_t> &occupied) {
  Vec3 delta{};
  double length_squared = 0.0;
  for (int d = 0; d < 3; ++d) {
    delta[d] = endpoint[d] - origin[d];
    length_squared += delta[d] * delta[d];
  }
  const double distance = std::sqrt(length_squared);
  if (!std::isfinite(distance))
    throw std::invalid_argument("LiDAR ray distance must be finite");

  Vec3 direction{};
  if (distance > 0.0)
    for (int d = 0; d < 3; ++d)
      direction[d] = delta[d] / distance;

  const bool has_return = distance <= max_distance;
  const double march_distance = std::min(distance, max_distance);
  if (march_distance > 0.0) {
    const auto cell = topology.extent_at(topology.max_depth);
    Vec3 next_crossing{};
    Vec3 crossing_step{};

    for (int d = 0; d < 3; ++d) {
      if (direction[d] == 0.0) {
        next_crossing[d] = std::numeric_limits<double>::infinity();
        crossing_step[d] = std::numeric_limits<double>::infinity();
        continue;
      }

      const double coordinate =
          (origin[d] - topology.origin[d]) / cell[d];
      const double boundary_index =
          direction[d] > 0.0 ? std::floor(coordinate) + 1.0
                             : std::ceil(coordinate) - 1.0;
      const double boundary =
          topology.origin[d] + boundary_index * cell[d];
      next_crossing[d] = (boundary - origin[d]) / direction[d];
      crossing_step[d] = cell[d] / std::fabs(direction[d]);
      if (next_crossing[d] <= 0.0)
        throw std::invalid_argument(
            "LiDAR ray could not initialize grid traversal");
    }

    auto mark_free_between = [&](double begin, double end) {
      adaptive::detail::DVec sample{};
      const double t = begin + (end - begin) * 0.5;
      for (int d = 0; d < 3; ++d)
        sample[d] = origin[d] + direction[d] * t;
      const int64_t voxel = topology.find_containing(sample);
      if (voxel >= 0)
        free[static_cast<size_t>(batch * topology.size() + voxel)] = 1;
    };

    double current = 0.0;
    int64_t traversed = 0;
    while (current < march_distance) {
      if (traversed >= max_voxels)
        throw std::length_error(
            "LiDAR ray traversal exceeds max_ray_voxels");
      const double next = std::min(
          march_distance,
          std::min(next_crossing[0],
                   std::min(next_crossing[1], next_crossing[2])));
      if (next > current)
        mark_free_between(current, next);
      ++traversed;
      if (next >= march_distance)
        break;

      for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(next_crossing[d]))
          continue;
        const double tolerance =
            8.0 * std::numeric_limits<double>::epsilon() *
            std::max(std::fabs(next), std::fabs(next_crossing[d]));
        if (next_crossing[d] <= next ||
            next_crossing[d] - next <= tolerance)
          next_crossing[d] += crossing_step[d];
      }
      current = next;
    }
  }

  if (has_return) {
    adaptive::detail::DVec hit{};
    for (int d = 0; d < 3; ++d)
      hit[d] = endpoint[d];
    const int64_t voxel = topology.find_containing(hit);
    if (voxel >= 0)
      occupied[static_cast<size_t>(batch * topology.size() + voxel)] = 1;
  }
}

} // namespace

at::Tensor LidarOccupancyTargets::labels() const {
  return occupied.to(at::kFloat);
}

LidarOccupancyTargets from_lidar(
    const at::Tensor &points, const at::Tensor &sensor_origins,
    const at::Tensor &ray_batch_ids, int64_t batch_size,
    const adaptive::AdaptiveSparseTensor &reference,
    const LidarRayOptions &options, const at::Tensor &sensor_to_world) {
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(std::isfinite(options.max_ray_distance) &&
                  options.max_ray_distance > 0.0,
              "max_ray_distance must be positive and finite");
  TORCH_CHECK(options.max_ray_voxels > 0,
              "max_ray_voxels must be positive");
  TORCH_CHECK(reference.dimension() == 3,
              "LiDAR target generation requires a 3D reference grid");
  TORCH_CHECK(reference.device().is_cpu(),
              "LiDAR target generation currently requires CPU reference "
              "features");
  require_finite_points(points, "points");
  require_finite_points(sensor_origins, "sensor_origins");

  const int64_t rays = points.size(0);
  TORCH_CHECK(sensor_origins.size(0) == rays,
              "points and sensor_origins must have the same number of rows");
  TORCH_CHECK(ray_batch_ids.device().is_cpu() &&
                  ray_batch_ids.scalar_type() == at::kLong &&
                  ray_batch_ids.dim() == 1 && ray_batch_ids.size(0) == rays,
              "ray_batch_ids must be a CPU int64 tensor with shape [R]");

  auto point_values = points.detach().to(at::kDouble).contiguous();
  auto origin_values = sensor_origins.detach().to(at::kDouble).contiguous();
  auto batch_values = ray_batch_ids.contiguous();
  const double *point_data = point_values.data_ptr<double>();
  const double *origin_data = origin_values.data_ptr<double>();
  const int64_t *batch_data = batch_values.data_ptr<int64_t>();

  at::Tensor pose_values;
  const double *pose_data = nullptr;
  if (sensor_to_world.defined()) {
    TORCH_CHECK(sensor_to_world.device().is_cpu() &&
                    sensor_to_world.dim() == 3 &&
                    sensor_to_world.size(0) == batch_size &&
                    sensor_to_world.size(1) == 4 &&
                    sensor_to_world.size(2) == 4,
                "sensor_to_world must have shape [B, 4, 4] on CPU");
    TORCH_CHECK(sensor_to_world.is_floating_point() &&
                    at::isfinite(sensor_to_world).all().item<bool>(),
                "sensor_to_world must be finite and floating point");
    pose_values = sensor_to_world.detach().to(at::kDouble).contiguous();
    pose_data = pose_values.data_ptr<double>();
  }

  TORCH_CHECK(reference.size() == 0 ||
                  batch_size <= std::numeric_limits<int64_t>::max() /
                                    reference.size(),
              "batch_size times voxel count overflows int64");
  const int64_t total = batch_size * reference.size();
  std::vector<uint8_t> free(static_cast<size_t>(total), 0);
  std::vector<uint8_t> occupied(static_cast<size_t>(total), 0);
  const auto topology = reference.topology();

  for (int64_t ray = 0; ray < rays; ++ray) {
    const int64_t batch = batch_data[ray];
    TORCH_CHECK(batch >= 0 && batch < batch_size,
                "ray_batch_ids contains an out-of-range batch index");
    Vec3 point = read_vec(point_data, ray);
    Vec3 origin = read_vec(origin_data, ray);
    if (pose_data) {
      point = transform_point(point, pose_data, batch);
      origin = transform_point(origin, pose_data, batch);
    }
    mark_ray(*topology, origin, point, options.max_ray_distance,
             options.max_ray_voxels, free, batch, occupied);
  }

  std::vector<uint8_t> observed(static_cast<size_t>(total), 0);
  for (size_t i = 0; i < observed.size(); ++i) {
    if (occupied[i])
      free[i] = 0;
    observed[i] = static_cast<uint8_t>(free[i] || occupied[i]);
  }

  const auto bool_options = at::TensorOptions().dtype(at::kBool);
  auto to_tensor = [&](const std::vector<uint8_t> &values) {
    if (values.empty())
      return at::empty({batch_size, reference.size()}, bool_options);
    return at::from_blob(const_cast<uint8_t *>(values.data()),
                         {batch_size, reference.size()}, bool_options)
        .clone();
  };
  return {to_tensor(occupied), to_tensor(free), to_tensor(observed)};
}

at::Tensor masked_binary_cross_entropy(
    const at::Tensor &predicted_occupancy,
    const LidarOccupancyTargets &targets, double epsilon) {
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0 && epsilon < 0.5,
              "epsilon must be finite and in (0, 0.5)");
  TORCH_CHECK(predicted_occupancy.dim() == 2 &&
                  predicted_occupancy.sizes() == targets.occupied.sizes(),
              "predicted_occupancy and labels must have matching [B, V] shapes");
  TORCH_CHECK(predicted_occupancy.device() == targets.occupied.device() &&
                  predicted_occupancy.device() == targets.observed.device() &&
                  predicted_occupancy.device() == targets.free.device(),
              "predictions and LiDAR targets must share a device");
  TORCH_CHECK(targets.occupied.scalar_type() == at::kBool &&
                  targets.free.scalar_type() == at::kBool &&
                  targets.observed.scalar_type() == at::kBool,
              "LiDAR target masks must be boolean tensors");
  TORCH_CHECK((targets.occupied | targets.free).equal(targets.observed),
              "observed mask must equal occupied OR free");
  TORCH_CHECK(!(targets.occupied & targets.free).any().item<bool>(),
              "a LiDAR voxel cannot be both occupied and free");
  TORCH_CHECK(predicted_occupancy.is_floating_point() &&
                  at::isfinite(predicted_occupancy).all().item<bool>(),
              "predicted occupancy must be finite and floating point");
  TORCH_CHECK((predicted_occupancy >= 0).all().item<bool>() &&
                  (predicted_occupancy <= 1).all().item<bool>(),
              "predicted occupancy must be in [0, 1]");

  const auto mask = targets.observed.to(predicted_occupancy.scalar_type());
  const auto labels =
      targets.occupied.to(predicted_occupancy.scalar_type());
  const auto probabilities = predicted_occupancy.clamp(epsilon, 1.0 - epsilon);
  const auto per_voxel =
      -(labels * probabilities.log() +
        (1.0 - labels) * at::log1p(-probabilities));
  const auto count = mask.sum();
  if (count.item<double>() == 0.0)
    return predicted_occupancy.sum() * 0.0;
  return (per_voxel * mask).sum() / count;
}

} // namespace occupancy
} // namespace riemann
