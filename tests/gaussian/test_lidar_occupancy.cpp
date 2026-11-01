#include "gaussian_voxel/occupancy.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <vector>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

using riemann::adaptive::AdaptiveSparseTensor;
using riemann::adaptive::AdaptiveVoxelConfig;
using namespace riemann::occupancy;

namespace {

AdaptiveSparseTensor make_grid() {
  std::vector<int64_t> coordinates;
  for (int64_t x = 0; x < 5; ++x)
    for (int64_t y = 0; y < 5; ++y)
      for (int64_t z = 0; z < 5; ++z)
        coordinates.insert(coordinates.end(), {x, y, z});
  const int64_t count = static_cast<int64_t>(coordinates.size() / 3);
  auto coords =
      at::from_blob(coordinates.data(), {count, 3}, at::kLong).clone();
  AdaptiveVoxelConfig config;
  config.base_voxel_size = {1.0, 1.0, 1.0};
  config.origin = {-2.0, -2.0, -2.0};
  return AdaptiveSparseTensor::from_voxels(
      coords, at::zeros({count}, at::kInt),
      at::zeros({count, 1}, at::kFloat), config);
}

int64_t voxel(const AdaptiveSparseTensor &grid, int64_t x, int64_t y,
              int64_t z) {
  return grid.index_of({x + 2, y + 2, z + 2}, 0);
}

} // namespace

TEST(lidar_dda_assigns_free_occupied_and_unknown) {
  auto grid = make_grid();
  auto points = at::tensor({1.2f, -1.5f, -1.5f}).reshape({1, 3});
  auto origins = at::tensor({-1.9f, -1.5f, -1.5f}).reshape({1, 3});
  auto batch_ids = at::zeros({1}, at::kLong);
  const auto targets =
      from_lidar(points, origins, batch_ids, 1, grid);

  CHECK(targets.observed[0][voxel(grid, -2, -2, -2)].item<bool>());
  CHECK(targets.free[0][voxel(grid, -2, -2, -2)].item<bool>());
  CHECK(targets.free[0][voxel(grid, -1, -2, -2)].item<bool>());
  CHECK(targets.free[0][voxel(grid, 0, -2, -2)].item<bool>());
  CHECK(targets.occupied[0][voxel(grid, 1, -2, -2)].item<bool>());
  CHECK(!targets.free[0][voxel(grid, 1, -2, -2)].item<bool>());
  CHECK(!targets.observed[0][voxel(grid, 2, -2, -2)].item<bool>());
  CHECK_EQ(targets.labels()[0][voxel(grid, 1, -2, -2)].item<float>(), 1.0f);
  CHECK_EQ(targets.labels()[0][voxel(grid, 0, -2, -2)].item<float>(), 0.0f);
}

TEST(lidar_dda_does_not_skip_short_cells_near_corner_crossings) {
  auto grid = make_grid();
  auto points = at::tensor({0.1f, 0.2f, -1.5f}).reshape({1, 3});
  auto origins = at::tensor({-1.9f, -1.89f, -1.5f}).reshape({1, 3});
  auto targets = from_lidar(points, origins, at::zeros({1}, at::kLong),
                            1, grid);

  CHECK(targets.free[0][voxel(grid, -2, -1, -2)].item<bool>());
}

TEST(lidar_batch_ids_keep_scene_targets_separate) {
  auto grid = make_grid();
  auto points =
      at::tensor({1.2f, -1.5f, -1.5f, -0.2f, 1.5f, -1.5f})
          .reshape({2, 3});
  auto origins =
      at::tensor({-1.9f, -1.5f, -1.5f, 0.1f, 1.5f, -1.5f})
          .reshape({2, 3});
  auto batch_ids = at::tensor({0, 1}, at::kLong);
  const auto targets =
      from_lidar(points, origins, batch_ids, 2, grid);

  const int64_t first_hit = voxel(grid, 1, -2, -2);
  const int64_t second_hit = voxel(grid, -1, 1, -2);
  CHECK(targets.occupied[0][first_hit].item<bool>());
  CHECK(!targets.observed[1][first_hit].item<bool>());
  CHECK(targets.occupied[1][second_hit].item<bool>());
  CHECK(!targets.observed[0][second_hit].item<bool>());
}

TEST(lidar_pose_transforms_sensor_points_to_world) {
  auto grid = make_grid();
  auto points = at::tensor({3.1f, 0.5f, 0.5f}).reshape({1, 3});
  auto origins = at::zeros({1, 3});
  auto batch_ids = at::zeros({1}, at::kLong);
  auto pose = at::eye(4).reshape({1, 4, 4});
  pose[0][0][3] = -2.0;
  pose[0][1][3] = -2.0;
  pose[0][2][3] = -2.0;
  const auto targets =
      from_lidar(points, origins, batch_ids, 1, grid, {}, pose);
  CHECK(targets.occupied[0][voxel(grid, 1, -2, -2)].item<bool>());
  CHECK(targets.free[0][voxel(grid, -2, -2, -2)].item<bool>());
}

TEST(lidar_max_distance_marks_free_without_fabricating_a_return) {
  auto grid = make_grid();
  auto points = at::tensor({10.0f, -1.5f, -1.5f}).reshape({1, 3});
  auto origins = at::tensor({-1.9f, -1.5f, -1.5f}).reshape({1, 3});
  auto batch_ids = at::zeros({1}, at::kLong);
  LidarRayOptions options;
  options.max_ray_distance = 2.0;
  const auto targets =
      from_lidar(points, origins, batch_ids, 1, grid, options);
  CHECK(targets.free[0][voxel(grid, -1, -2, -2)].item<bool>());
  CHECK(targets.free[0][voxel(grid, 0, -2, -2)].item<bool>());
  CHECK(!targets.occupied.any().item<bool>());
  CHECK(!targets.observed[0][voxel(grid, 2, -2, -2)].item<bool>());
}

TEST(lidar_ray_work_limit_and_invalid_inputs_are_reported) {
  auto grid = make_grid();
  auto points = at::tensor({1.2f, -1.5f, -1.5f}).reshape({1, 3});
  auto origins = at::tensor({-1.9f, -1.5f, -1.5f}).reshape({1, 3});
  auto batch_ids = at::zeros({1}, at::kLong);
  LidarRayOptions options;
  options.max_ray_voxels = 1;
  CHECK_THROWS(from_lidar(points, origins, batch_ids, 1, grid, options));
  CHECK_THROWS(from_lidar(points, origins, batch_ids, 0, grid));
  CHECK_THROWS(from_lidar(points, origins, at::zeros({1}, at::kInt), 1, grid));
}

TEST(lidar_masked_bce_ignores_unknown_and_backpropagates) {
  auto grid = make_grid();
  auto points = at::tensor({1.2f, -1.5f, -1.5f}).reshape({1, 3});
  auto origins = at::tensor({-1.9f, -1.5f, -1.5f}).reshape({1, 3});
  auto targets = from_lidar(points, origins, at::zeros({1}, at::kLong), 1,
                            grid);
  auto predicted =
      at::full({1, grid.size()}, 0.4f).set_requires_grad(true);
  const auto loss = masked_binary_cross_entropy(predicted, targets);
  CHECK(loss.item<float>() > 0.0f);
  loss.backward();
  CHECK(predicted.grad().defined());
  CHECK(predicted.grad().abs().sum().item<float>() > 0.0f);
  CHECK(predicted.grad()[0][voxel(grid, 2, -2, -2)].item<float>() == 0.0f);

  LidarOccupancyTargets empty{
      at::zeros({1, grid.size()}, at::kBool),
      at::zeros({1, grid.size()}, at::kBool),
      at::zeros({1, grid.size()}, at::kBool)};
  auto no_observation = at::full({1, grid.size()}, 0.5f)
                            .set_requires_grad(true);
  const auto zero = masked_binary_cross_entropy(no_observation, empty);
  CHECK_EQ(zero.item<float>(), 0.0f);
  zero.backward();
  CHECK_EQ(no_observation.grad().abs().sum().item<float>(), 0.0f);
}

TEST(lidar_empty_rays_leave_every_voxel_unknown) {
  auto grid = make_grid();
  auto targets = from_lidar(at::empty({0, 3}), at::empty({0, 3}),
                            at::empty({0}, at::kLong), 1, grid);
  CHECK(!targets.observed.any().item<bool>());
}
