#include "gaussian_voxel/voxelize.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <functional>
#include <vector>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

using riemann::adaptive::AdaptiveSparseTensor;
using riemann::adaptive::AdaptiveVoxelConfig;
using riemann::gaussian::GaussianTensor;
using namespace riemann::gaussian_voxel;

namespace {

AdaptiveSparseTensor make_uniform_grid(int64_t lower, int64_t upper) {
  std::vector<int64_t> coordinates;
  for (int64_t x = lower; x <= upper; ++x)
    for (int64_t y = lower; y <= upper; ++y)
      for (int64_t z = lower; z <= upper; ++z) {
        coordinates.push_back(x);
        coordinates.push_back(y);
        coordinates.push_back(z);
      }

  const int64_t count = static_cast<int64_t>(coordinates.size() / 3);
  auto coords = count == 0
                    ? at::empty({0, 3}, at::kLong)
                    : at::from_blob(coordinates.data(), {count, 3}, at::kLong)
                          .clone();
  auto levels = at::zeros({count}, at::kInt);
  auto features = at::zeros({count, 1}, at::kDouble);

  AdaptiveVoxelConfig config;
  config.base_voxel_size = {1.0, 1.0, 1.0};
  config.origin = {static_cast<double>(lower), static_cast<double>(lower),
                   static_cast<double>(lower)};
  return AdaptiveSparseTensor::from_voxels(coords, levels, features, config);
}

GaussianTensor make_gaussians(const at::Tensor &positions,
                              const at::Tensor &rotations,
                              const at::Tensor &scales,
                              const at::Tensor &opacities,
                              const at::Tensor &features) {
  return GaussianTensor(positions, rotations, scales, opacities,
                        at::zeros({positions.size(0), 3}, positions.options()),
                        features);
}

at::Tensor objective_tensor(const GaussianTensor &gaussians,
                            const AdaptiveSparseTensor &grid,
                            const at::Tensor &feature_probe,
                            const at::Tensor &density_probe) {
  const auto output = rasterize(gaussians, grid);
  return (output.voxels.features() * feature_probe).sum() +
         (output.density * density_probe).sum();
}

double objective(const GaussianTensor &gaussians,
                 const AdaptiveSparseTensor &grid,
                 const at::Tensor &feature_probe,
                 const at::Tensor &density_probe) {
  return objective_tensor(gaussians, grid, feature_probe, density_probe)
      .item<double>();
}

at::Tensor numerical_gradient(
    const std::function<double(const at::Tensor &)> &loss,
    at::Tensor values, double epsilon = 1e-6) {
  auto result = at::zeros_like(values);
  at::NoGradGuard no_grad;
  auto flat = values.reshape({-1});
  auto grad_flat = result.reshape({-1});
  for (int64_t i = 0; i < flat.numel(); ++i) {
    const double original = flat[i].item<double>();
    flat[i] = original + epsilon;
    const double upper = loss(values);
    flat[i] = original - epsilon;
    const double lower = loss(values);
    flat[i] = original;
    grad_flat[i] = (upper - lower) / (2.0 * epsilon);
  }
  return result;
}

AdaptiveSparseTensor make_mixed_grid() {
  std::vector<int64_t> coords{0, 0, 0};
  std::vector<int32_t> levels{0};
  for (int64_t x : {2, 3})
    for (int64_t y : {0, 1})
      for (int64_t z : {0, 1}) {
        coords.insert(coords.end(), {x, y, z});
        levels.push_back(1);
      }

  auto coordinates =
      at::from_blob(coords.data(),
                    {static_cast<int64_t>(levels.size()), 3}, at::kLong)
          .clone();
  auto level_tensor =
      at::from_blob(levels.data(), {static_cast<int64_t>(levels.size())},
                    at::kInt)
          .clone();
  AdaptiveVoxelConfig config;
  config.base_voxel_size = {4.0, 4.0, 4.0};
  config.max_depth = 1;
  return AdaptiveSparseTensor::from_voxels(
      coordinates, level_tensor,
      at::zeros({static_cast<int64_t>(levels.size()), 1}, at::kDouble),
      config);
}

} // namespace

TEST(gaussian_voxel_single_gaussian_density_and_features) {
  auto grid = make_uniform_grid(-2, 2);
  auto positions = at::tensor({0.2, 0.1, -0.1}, at::kDouble).reshape({1, 3});
  auto rotations =
      at::tensor({1.0, 0.0, 0.0, 0.0}, at::kDouble).reshape({1, 4});
  auto scales = at::full({1, 3}, 0.4, at::kDouble);
  auto opacities = at::full({1, 1}, 0.7, at::kDouble);
  auto features = at::tensor({2.0, -3.0}, at::kDouble).reshape({1, 2});
  auto gaussians =
      make_gaussians(positions, rotations, scales, opacities, features);

  const auto result = rasterize(gaussians, grid);
  CHECK_EQ(result.voxels.size(), grid.size());
  CHECK_EQ(result.voxels.channels(), 2);
  CHECK(result.density.sizes() == at::IntArrayRef({grid.size(), 1}));
  CHECK(result.occupancy.sizes() == at::IntArrayRef({grid.size(), 1}));
  CHECK(result.density.sum().item<double>() > 0.0);
  CHECK((result.occupancy >= 0).all().item<bool>());
  CHECK((result.occupancy < 1).all().item<bool>());

  const auto active = result.density.squeeze(1) > 0;
  const auto expected = features.expand({active.sum().item<int64_t>(), 2});
  CHECK_ALLCLOSE(result.voxels.features().index({active}), expected, 1e-12,
                 1e-12);
  CHECK_ALLCLOSE(result.occupancy, -at::expm1(-result.density), 1e-12, 1e-12);
}

TEST(gaussian_voxel_weighted_mean_and_confidence) {
  auto grid = make_uniform_grid(-2, 2);
  auto positions = at::zeros({2, 3}, at::kDouble);
  auto rotations = at::tensor({1.0, 0.0, 0.0, 0.0}, at::kDouble)
                       .reshape({1, 4})
                       .repeat({2, 1});
  auto scales = at::full({2, 3}, 0.4, at::kDouble);
  auto opacity = at::tensor({0.5, 0.25}, at::kDouble).reshape({2, 1});
  auto features = at::tensor({2.0, 6.0}, at::kDouble).reshape({2, 1});
  auto confidence =
      at::tensor({1.0, 2.0}, at::kDouble).set_requires_grad(true);
  auto gaussians = make_gaussians(positions, rotations, scales, opacity,
                                  features);

  const auto result = rasterize(gaussians, grid, {}, confidence);
  const auto active = result.density.squeeze(1) > 0;
  CHECK(active.any().item<bool>());
  CHECK_ALLCLOSE(result.voxels.features().index({active}),
                 at::full({active.sum().item<int64_t>(), 1}, 4.0, at::kDouble),
                 1e-12, 1e-12);
  (result.voxels.features().sum() + result.density.sum()).backward();
  CHECK(confidence.grad().defined());
  CHECK(confidence.grad().abs().sum().item<double>() > 0.0);

  GaussianVoxelOptions linear_options;
  linear_options.occupancy_model = OccupancyModel::ClampedLinear;
  const auto linear = rasterize(gaussians, grid, linear_options, confidence);
  CHECK_ALLCLOSE(linear.occupancy, linear.density.clamp(0.0, 1.0), 1e-12,
                 1e-12);

  GaussianVoxelOptions sum_options;
  sum_options.feature_aggregation = FeatureAggregation::WeightedSum;
  const auto sum = rasterize(gaussians, grid, sum_options, confidence.detach());
  CHECK_ALLCLOSE(sum.voxels.features(),
                 result.voxels.features() * result.density, 1e-12, 1e-12);
}

TEST(gaussian_voxel_empty_inputs_and_invalid_scales) {
  auto grid = make_uniform_grid(-1, 1);
  auto empty = GaussianTensor::empty(0, at::TensorOptions().dtype(at::kDouble),
                                     3);
  const auto empty_result = rasterize(empty, grid);
  CHECK_EQ(empty_result.density.sum().item<double>(), 0.0);
  CHECK_EQ(empty_result.voxels.features().abs().sum().item<double>(), 0.0);

  auto no_voxels = make_uniform_grid(1, 0);
  const auto no_voxel_result = rasterize(empty, no_voxels);
  CHECK_EQ(no_voxel_result.density.numel(), 0);
  CHECK_EQ(no_voxel_result.voxels.features().numel(), 0);

  auto positions = at::zeros({1, 3}, at::kDouble);
  auto rotations = at::tensor({1.0, 0.0, 0.0, 0.0}, at::kDouble).reshape({1, 4});
  auto scales = at::tensor({1.0, 0.0, 1.0}, at::kDouble).reshape({1, 3});
  auto invalid =
      make_gaussians(positions, rotations, scales, at::ones({1, 1}, at::kDouble),
                     at::ones({1, 1}, at::kDouble));
  CHECK_THROWS(rasterize(invalid, grid));
}

TEST(gaussian_voxel_support_lookup_respects_association_limit) {
  auto grid = make_uniform_grid(-2, 2);
  auto gaussians = make_gaussians(
      at::zeros({1, 3}, at::kDouble),
      at::tensor({1.0, 0.0, 0.0, 0.0}, at::kDouble).reshape({1, 4}),
      at::full({1, 3}, 0.4, at::kDouble),
      at::ones({1, 1}, at::kDouble),
      at::ones({1, 1}, at::kDouble));
  GaussianVoxelOptions options;
  options.max_associations = 1;
  CHECK_THROWS(rasterize(gaussians, grid, options));
}

TEST(gaussian_voxel_float32_reference) {
  auto grid = make_uniform_grid(-1, 1);
  auto gaussians = make_gaussians(
      at::zeros({1, 3}, at::kFloat),
      at::tensor({1.0f, 0.0f, 0.0f, 0.0f}, at::kFloat).reshape({1, 4}),
      at::full({1, 3}, 0.4f, at::kFloat),
      at::full({1, 1}, 0.8f, at::kFloat),
      at::full({1, 2}, 0.25f, at::kFloat));
  const auto output = rasterize(gaussians, grid);
  CHECK(output.density.scalar_type() == at::kFloat);
  CHECK(at::isfinite(output.density).all().item<bool>());
  CHECK(output.density.sum().item<float>() > 0.0f);
  CHECK_ALLCLOSE(output.voxels.features().sum(),
                 at::tensor(0.5f, at::kFloat) *
                     (output.density > 0).sum().to(at::kFloat),
                 1e-5, 1e-6);
}

TEST(gaussian_voxel_mixed_resolution_accumulates_each_leaf_once) {
  auto grid = make_mixed_grid();
  auto positions = at::tensor({3.9, 2.0, 2.0}, at::kDouble).reshape({1, 3});
  auto rotations =
      at::tensor({1.0, 0.0, 0.0, 0.0}, at::kDouble).reshape({1, 4});
  auto scales = at::full({1, 3}, 0.4, at::kDouble);
  auto gaussians =
      make_gaussians(positions, rotations, scales,
                     at::ones({1, 1}, at::kDouble),
                     at::ones({1, 1}, at::kDouble));
  const auto result = rasterize(gaussians, grid);
  const auto coarse = grid.levels() == 0;
  const auto fine = grid.levels() == 1;
  CHECK(result.density.squeeze(1).index({coarse}).sum().item<double>() > 0.0);
  CHECK(result.density.squeeze(1).index({fine}).sum().item<double>() > 0.0);
  CHECK(!grid.has_overlapping_leaves());
}

TEST(gaussian_voxel_parameter_gradients_match_finite_differences) {
  auto grid = make_uniform_grid(-2, 2);
  auto base_positions =
      at::tensor({0.17, 0.12, -0.08}, at::kDouble).reshape({1, 3});
  auto base_rotations =
      at::tensor({0.91, 0.13, 0.21, 0.31}, at::kDouble).reshape({1, 4});
  auto base_scales =
      at::tensor({0.35, 0.52, 0.43}, at::kDouble).reshape({1, 3});
  auto base_opacities = at::tensor({0.63}, at::kDouble).reshape({1, 1});
  auto base_features = at::tensor({0.4, -0.7}, at::kDouble).reshape({1, 2});
  auto feature_probe = at::randn({grid.size(), 2}, at::kDouble);
  auto density_probe = at::randn({grid.size(), 1}, at::kDouble);

  auto positions = base_positions.clone().set_requires_grad(true);
  auto rotations = base_rotations.clone().set_requires_grad(true);
  auto scales = base_scales.clone().set_requires_grad(true);
  auto opacities = base_opacities.clone().set_requires_grad(true);
  auto features = base_features.clone().set_requires_grad(true);
  auto gaussians =
      make_gaussians(positions, rotations, scales, opacities, features);
  objective_tensor(gaussians, grid, feature_probe, density_probe).backward();

  auto compare_gradient = [&](const at::Tensor &actual,
                              const at::Tensor &base,
                              const std::function<GaussianTensor(
                                  const at::Tensor &)> &replace) {
    const auto numeric = numerical_gradient(
        [&](const at::Tensor &value) {
          return objective(replace(value), grid, feature_probe, density_probe);
        },
        base.clone());
    CHECK_ALLCLOSE(actual.grad(), numeric, 2e-5, 1e-6);
  };
  compare_gradient(positions, base_positions, [&](const at::Tensor &value) {
    return make_gaussians(value, base_rotations, base_scales, base_opacities,
                          base_features);
  });
  compare_gradient(rotations, base_rotations, [&](const at::Tensor &value) {
    return make_gaussians(base_positions, value, base_scales, base_opacities,
                          base_features);
  });
  compare_gradient(scales, base_scales, [&](const at::Tensor &value) {
    return make_gaussians(base_positions, base_rotations, value, base_opacities,
                          base_features);
  });
  compare_gradient(opacities, base_opacities, [&](const at::Tensor &value) {
    return make_gaussians(base_positions, base_rotations, base_scales, value,
                          base_features);
  });
  compare_gradient(features, base_features, [&](const at::Tensor &value) {
    return make_gaussians(base_positions, base_rotations, base_scales,
                          base_opacities, value);
  });
}

TEST(gaussian_voxel_cuda_inputs_fail_explicitly) {
#ifdef USE_CUDA
  if (!at::cuda::is_available())
    throw riemann_test::Skip("CUDA runtime or device is unavailable");
  auto gaussians = GaussianTensor::empty(
      0, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA), 1);
  auto grid = make_uniform_grid(-1, 1);
  CHECK_THROWS(rasterize(gaussians, grid));
#endif
}
