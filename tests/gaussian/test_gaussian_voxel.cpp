#include "gaussian_voxel/voxelize.hpp"
#include "gaussian_voxel/refine.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <functional>
#include <cmath>
#include <vector>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
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

TEST(gaussian_voxel_cuda_forward_backward_matches_cpu) {
#ifdef USE_CUDA
  if (!at::cuda::is_available())
    throw riemann_test::Skip("CUDA runtime or device is unavailable");
  // A nondefault stream catches launches accidentally using the default stream.
  const c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
  for (auto dtype : {at::kFloat, at::kDouble})
    for (auto aggregation : {FeatureAggregation::WeightedSum,
                             FeatureAggregation::NormalizedWeightedMean})
      for (auto model : {OccupancyModel::Poisson, OccupancyModel::ClampedLinear}) {
        auto grid = make_mixed_grid();
        const auto opts = at::TensorOptions().dtype(dtype);
        std::vector<at::Tensor> cpu{
            at::tensor({1.2, 1.4, 1.7, 3.8, 2.2, 1.4}, opts).reshape({2, 3}),
            at::tensor({1., .2, -.1, .3, 1., -.3, .1, .2}, opts).reshape({2, 4}),
            at::tensor({.7, .8, .9, .8, .6, .9}, opts).reshape({2, 3}),
            at::tensor({.7, .4}, opts).reshape({2, 1}),
            at::arange(10, opts).reshape({2, 5}) / 10,
            at::tensor({.8, 0.}, opts)};
        std::vector<at::Tensor> gpu;
        for (auto &value : cpu) {
          value.set_requires_grad(true);
          gpu.push_back(value.detach().to(at::kCUDA).set_requires_grad(true));
        }
        GaussianVoxelOptions options;
        options.feature_aggregation = aggregation;
        options.occupancy_model = model;
        auto left = rasterize(make_gaussians(cpu[0], cpu[1], cpu[2], cpu[3], cpu[4]),
                              grid, options, cpu[5]);
        auto right = rasterize(make_gaussians(gpu[0], gpu[1], gpu[2], gpu[3], gpu[4]),
                               grid.to(at::kCUDA), options, gpu[5]);
        const double tolerance = dtype == at::kFloat ? 2e-4 : 2e-10;
        CHECK_ALLCLOSE(right.density.cpu(), left.density, tolerance, tolerance);
        CHECK_ALLCLOSE(right.occupancy.cpu(), left.occupancy, tolerance, tolerance);
        CHECK_ALLCLOSE(right.voxels.features().cpu(), left.voxels.features(),
                       tolerance, tolerance);
        auto loss = [](const GaussianVoxelOutput &output) {
          return output.voxels.features().square().sum() +
                 output.density.square().sum() + output.occupancy.sum();
        };
        loss(left).backward();
        loss(right).backward();
        for (size_t i = 0; i < cpu.size(); ++i)
          CHECK_ALLCLOSE(gpu[i].grad().cpu(), cpu[i].grad(), tolerance, tolerance);
        options.max_associations = 1;
        CHECK_THROWS(rasterize(make_gaussians(gpu[0], gpu[1], gpu[2], gpu[3], gpu[4]),
                               grid, options));
      }
#else
  throw riemann_test::Skip("CUDA kernels are not compiled in this build");
#endif
}

TEST(gaussian_voxel_empty_support_keeps_zero_gradients) {
  for (auto device : {at::Device(at::kCPU)
#ifdef USE_CUDA
                      , at::Device(at::kCUDA)
#endif
       }) {
#ifdef USE_CUDA
    if (device.is_cuda() && !at::cuda::is_available())
      continue;
#endif
    auto options = at::TensorOptions().dtype(at::kDouble).device(device);
    auto positions = at::full({1, 3}, 100., options).set_requires_grad(true);
    auto rotations = at::tensor({1., 0., 0., 0.}, options).reshape({1, 4});
    auto scales = at::ones({1, 3}, options);
    auto opacity = at::ones({1, 1}, options).set_requires_grad(true);
    auto features = at::empty({1, 0}, options).set_requires_grad(true);
    auto result = rasterize(make_gaussians(positions, rotations, scales, opacity,
                                          features), make_uniform_grid(-1, 1));
    result.density.sum().backward();
    CHECK_EQ(result.density.sum().item<double>(), 0.);
    CHECK_EQ(positions.grad().abs().sum().item<double>(), 0.);
    CHECK_EQ(opacity.grad().abs().sum().item<double>(), 0.);
    auto empty = GaussianTensor::empty(0, options, 0);
    CHECK_EQ(rasterize(empty, make_uniform_grid(-1, 1)).density.sum().item<double>(), 0.);
  }
}

TEST(gaussian_voxel_adaptive_refinement_obeys_budgets_and_preserves_input) {
  auto coords = at::tensor({0, 0, 0, 1, 0, 0}, at::kLong).reshape({2, 3});
  AdaptiveVoxelConfig config;
  config.base_voxel_size = {1., 1., 1.};
  config.max_depth = 3;
  auto grid = AdaptiveSparseTensor::from_voxels(
      coords, at::zeros({2}, at::kInt), at::zeros({2, 1}, at::kDouble), config);
  const auto version = grid.topology_version();
  auto positions = at::tensor({.5, .5, .5}, at::kDouble).reshape({1, 3});
  positions.set_requires_grad(true);
  auto gaussians = make_gaussians(
      positions, at::tensor({1., 0., 0., 0.}, at::kDouble).reshape({1, 4}),
      at::full({1, 3}, .3, at::kDouble), at::ones({1, 1}, at::kDouble),
      at::ones({1, 2}, at::kDouble));
  AdaptiveRefinementOptions refine;
  refine.criterion = RefinementCriterion::Density;
  refine.score_threshold = .01;
  refine.max_depth = 3;
  refine.max_voxels = 16;
  refine.max_splits_per_iteration = 1;
  refine.max_iterations = 10;
  auto result = rasterize_adaptive(gaussians, grid, {}, refine);
  CHECK_EQ(result.rasterized.voxels.size(), 16);
  CHECK(result.leaf_counts == std::vector<int64_t>({2, 9, 16}));
  CHECK(result.stop_reason == "max_voxels");
  CHECK(!result.rasterized.voxels.has_overlapping_leaves());
  CHECK_EQ(grid.topology_version(), version);
  CHECK_EQ(grid.size(), 2);
  CHECK_ALLCLOSE(result.rasterized.voxels.volumes().sum(), grid.volumes().sum(), 1e-12, 1e-12);
  auto repeat = rasterize_adaptive(gaussians, grid, {}, refine);
  CHECK(at::equal(result.rasterized.voxels.coordinates(), repeat.rasterized.voxels.coordinates()));
  result.rasterized.density.sum().backward();
  CHECK(at::isfinite(positions.grad()).all().item<bool>());
  CHECK(positions.grad().abs().sum().item<double>() > 0);
  refine.max_voxels = 1;
  CHECK_THROWS(rasterize_adaptive(gaussians, grid, {}, refine));
  refine.max_voxels = 100;
  refine.max_depth = 4;
  CHECK_THROWS(rasterize_adaptive(gaussians, grid, {}, refine));
}

TEST(gaussian_voxel_refinement_threshold_depth_and_iteration_stops) {
  AdaptiveVoxelConfig config;
  config.base_voxel_size = {1., 1., 1.};
  config.max_depth = 1;
  auto grid = AdaptiveSparseTensor::from_voxels(
      at::zeros({1, 3}, at::kLong), at::zeros({1}, at::kInt),
      at::zeros({1, 1}, at::kDouble), config);
  auto gaussians = make_gaussians(
      at::full({1, 3}, .5, at::kDouble),
      at::tensor({1., 0., 0., 0.}, at::kDouble).reshape({1, 4}),
      at::ones({1, 3}, at::kDouble), at::ones({1, 1}, at::kDouble),
      at::empty({1, 0}, at::kDouble));
  AdaptiveRefinementOptions refine;
  refine.max_depth = 1;
  refine.score_threshold = 1;
  CHECK(rasterize_adaptive(gaussians, grid, {}, refine).stop_reason == "threshold");
  refine.score_threshold = 0;
  refine.max_iterations = 1;
  auto one = rasterize_adaptive(gaussians, grid, {}, refine);
  CHECK_EQ(one.rasterized.voxels.size(), 8);
  CHECK(one.stop_reason == "max_iterations");
  refine.max_iterations = 2;
  CHECK(rasterize_adaptive(gaussians, grid, {}, refine).stop_reason == "max_depth");
  refine.max_iterations = 0;
  CHECK_EQ(rasterize_adaptive(gaussians, grid, {}, refine).rasterized.voxels.size(), 1);
}

TEST(gaussian_voxel_cuda_adaptive_topology_matches_cpu) {
#ifdef USE_CUDA
  if (!at::cuda::is_available())
    throw riemann_test::Skip("CUDA runtime or device is unavailable");
  const c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
  auto grid = make_mixed_grid();
  auto positions = at::tensor({1.21, 1.43, 1.71}, at::kDouble).reshape({1, 3});
  auto rotations = at::tensor({1., .2, -.1, .3}, at::kDouble).reshape({1, 4});
  auto scales = at::full({1, 3}, .7, at::kDouble);
  auto opacity = at::full({1, 1}, .7, at::kDouble);
  auto features = at::full({1, 3}, 2., at::kDouble);
  AdaptiveRefinementOptions refine;
  refine.max_depth = 1;
  refine.max_voxels = grid.size() + 7;
  refine.max_iterations = 3;
  refine.score_threshold = 0;
  auto left = rasterize_adaptive(
      make_gaussians(positions, rotations, scales, opacity, features), grid, {}, refine);
  auto right = rasterize_adaptive(
      make_gaussians(positions.cuda(), rotations.cuda(), scales.cuda(),
                     opacity.cuda(), features.cuda()), grid, {}, refine);
  CHECK(left.leaf_counts == right.leaf_counts);
  CHECK(left.stop_reason == right.stop_reason);
  CHECK(at::equal(left.rasterized.voxels.coordinates(), right.rasterized.voxels.coordinates()));
  CHECK(at::equal(left.rasterized.voxels.levels(), right.rasterized.voxels.levels()));
  CHECK_ALLCLOSE(right.rasterized.density.cpu(), left.rasterized.density, 1e-10, 1e-10);
#else
  throw riemann_test::Skip("CUDA kernels are not compiled in this build");
#endif
}

TEST(gaussian_voxel_cuda_density_only_and_total_pair_limit) {
#ifdef USE_CUDA
  if (!at::cuda::is_available())
    throw riemann_test::Skip("CUDA runtime or device is unavailable");
  AdaptiveVoxelConfig config;
  config.base_voxel_size = {1., 1., 1.};
  auto grid = AdaptiveSparseTensor::from_voxels(
      at::zeros({1, 3}, at::kLong), at::zeros({1}, at::kInt),
      at::zeros({1, 1}), config);
  const auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kDouble);
  auto positions = at::full({2, 3}, .41, options).set_requires_grad(true);
  auto gaussians = make_gaussians(
      positions, at::tensor({1., 0., 0., 0.}, options).reshape({1, 4}).repeat({2, 1}),
      at::full({2, 3}, .1, options), at::full({2, 1}, .5, options),
      at::empty({2, 0}, options));
  auto output = rasterize(gaussians, grid);
  CHECK_EQ(output.voxels.channels(), 0);
  output.density.sum().backward();
  CHECK(at::isfinite(positions.grad()).all().item<bool>());
  CHECK(positions.grad().abs().sum().item<double>() > 0);
  const auto second_order = rasterize(gaussians, grid).density.sum();
  CHECK_THROWS(torch::autograd::grad(
      {second_order}, {positions}, {}, false, true));
  GaussianVoxelOptions bounded;
  bounded.max_associations = 1; // Each support fits; the combined pair count does not.
  CHECK_THROWS(rasterize(gaussians, grid, bounded));
#else
  throw riemann_test::Skip("CUDA kernels are not compiled in this build");
#endif
}

TEST(gaussian_voxel_refinement_improves_midpoint_integration) {
  AdaptiveVoxelConfig config;
  config.base_voxel_size = {1., 1., 1.};
  config.max_depth = 2;
  auto grid = AdaptiveSparseTensor::from_voxels(
      at::zeros({1, 3}, at::kLong), at::zeros({1}, at::kInt),
      at::zeros({1, 1}, at::kDouble), config);
  auto gaussians = make_gaussians(
      at::full({1, 3}, .5, at::kDouble),
      at::tensor({1., 0., 0., 0.}, at::kDouble).reshape({1, 4}),
      at::full({1, 3}, .15, at::kDouble), at::ones({1, 1}, at::kDouble),
      at::empty({1, 0}, at::kDouble));
  GaussianVoxelOptions options;
  options.support_radius = 4;
  AdaptiveRefinementOptions refine;
  refine.criterion = RefinementCriterion::Density;
  refine.score_threshold = 0;
  refine.min_density = 0;
  refine.max_voxels = 64;
  const double exact = std::pow(std::erf(.5 / (.15 * std::sqrt(2.))), 3);
  const double initial = rasterize(gaussians, grid, options).density.sum().item<double>();
  const auto refined = rasterize_adaptive(gaussians, grid, options, refine);
  const double actual = refined.rasterized.density.sum().item<double>();
  CHECK(refined.leaf_counts == std::vector<int64_t>({1, 8, 64}));
  CHECK(std::fabs(actual - exact) < .005);
  CHECK(std::fabs(actual - exact) < std::fabs(initial - exact) / 100);
  options.max_associations = 30; // Initial support fits; finer support does not.
  CHECK_THROWS(rasterize_adaptive(gaussians, grid, options, refine));
  CHECK_EQ(grid.size(), 1);
}
