#include "gaussian_voxel/voxelize.hpp"

#include "octree/tree.hpp"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riemann {
namespace gaussian_voxel {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

void validate_options(const GaussianVoxelOptions &options) {
  TORCH_CHECK(std::isfinite(options.support_radius) &&
                  options.support_radius > 0.0,
              "support_radius must be positive and finite");
  TORCH_CHECK(std::isfinite(options.opacity_density_scale) &&
                  options.opacity_density_scale >= 0.0,
              "opacity_density_scale must be nonnegative and finite");
  TORCH_CHECK(options.max_associations > 0,
              "max_associations must be positive");
  switch (options.feature_aggregation) {
  case FeatureAggregation::WeightedSum:
  case FeatureAggregation::NormalizedWeightedMean:
    break;
  default:
    TORCH_CHECK(false, "unknown feature aggregation mode");
  }
  switch (options.occupancy_model) {
  case OccupancyModel::Poisson:
  case OccupancyModel::ClampedLinear:
    break;
  default:
    TORCH_CHECK(false, "unknown occupancy model");
  }
}

void validate_cpu_gaussians(const gaussian::GaussianTensor &gaussians) {
  TORCH_CHECK(gaussians.device().is_cpu(),
              "Gaussian-to-voxel reference rasterization currently supports "
              "CPU tensors only; CUDA execution is not implemented yet");
  const auto &scales = gaussians.scales();
  TORCH_CHECK(at::isfinite(scales).all().item<bool>() &&
                  (scales > 0).all().item<bool>(),
              "Gaussian scales must be positive and finite");
  const auto &opacities = gaussians.opacities();
  TORCH_CHECK(at::isfinite(opacities).all().item<bool>() &&
                  (opacities >= 0).all().item<bool>() &&
                  (opacities <= 1).all().item<bool>(),
              "Gaussian opacities must be finite and in [0, 1]");
}

at::Tensor confidence_or_ones(const at::Tensor &confidence,
                              const gaussian::GaussianTensor &gaussians) {
  if (!confidence.defined())
    return at::ones({gaussians.size(), 1}, gaussians.positions().options());

  TORCH_CHECK(confidence.device().is_cpu(),
              "confidence must be a CPU tensor for CPU reference rasterization");
  TORCH_CHECK(confidence.scalar_type() == gaussians.dtype(),
              "confidence and Gaussian tensors must use the same dtype");
  TORCH_CHECK((confidence.dim() == 1 && confidence.size(0) == gaussians.size()) ||
                  (confidence.dim() == 2 && confidence.size(0) == gaussians.size() &&
                   confidence.size(1) == 1),
              "confidence must have shape [N] or [N, 1]");
  auto result = confidence.reshape({gaussians.size(), 1});
  TORCH_CHECK(at::isfinite(result).all().item<bool>() &&
                  (result >= 0).all().item<bool>(),
              "confidence values must be finite and nonnegative");
  return result;
}

void check_support_work(const adaptive::detail::Topology &topology,
                        const adaptive::detail::DVec &lower,
                        const adaptive::detail::DVec &upper,
                        int64_t limit) {
  int64_t total = 0;
  for (uint32_t level : topology.present_levels) {
    const auto extent = topology.extent_at(level);
    int64_t cells = 1;
    for (int d = 0; d < topology.D; ++d) {
      const double a = (lower[d] - topology.origin[d]) / extent[d];
      const double b = (upper[d] - topology.origin[d]) / extent[d];
      TORCH_CHECK(std::isfinite(a) && std::isfinite(b) &&
                      std::fabs(a) <= 4e18 && std::fabs(b) <= 4e18,
                  "Gaussian support coordinates are out of range");
      const int64_t first = static_cast<int64_t>(std::floor(a + 1e-9));
      const int64_t last = static_cast<int64_t>(std::ceil(b - 1e-9)) - 1;
      if (last < first) {
        cells = 0;
        break;
      }
      const uint64_t axis_cells =
          static_cast<uint64_t>(last - first) + 1;
      TORCH_CHECK(axis_cells <= static_cast<uint64_t>(limit / cells),
                  "Gaussian support lookup exceeds max_associations");
      cells *= static_cast<int64_t>(axis_cells);
    }
    TORCH_CHECK(cells <= limit - total,
                "Gaussian support lookup exceeds max_associations");
    total += cells;
  }
}

struct Associations {
  std::vector<int64_t> gaussian;
  std::vector<int64_t> voxel;
};

Associations find_associations(const gaussian::GaussianTensor &gaussians,
                               const adaptive::AdaptiveSparseTensor &reference,
                               const GaussianVoxelOptions &options) {
  const auto topology = reference.topology();
  const auto positions =
      gaussians.positions().detach().to(at::kDouble).contiguous();
  const auto covariances =
      gaussians.covariances().detach().to(at::kDouble).contiguous();
  const double *means = positions.data_ptr<double>();
  const double *cov = covariances.data_ptr<double>();
  Associations result;

  for (int64_t g = 0; g < gaussians.size(); ++g) {
    adaptive::detail::DVec lower{}, upper{};
    for (int d = 0; d < 3; ++d) {
      const double mean = static_cast<double>(means[g * 3 + d]);
      const double variance = static_cast<double>(cov[g * 9 + d * 3 + d]);
      TORCH_CHECK(std::isfinite(mean) && std::isfinite(variance) &&
                      variance > 0.0,
                  "Gaussian means and covariance diagonals must be finite, "
                  "with positive variance");
      const double reach = options.support_radius * std::sqrt(variance);
      lower[d] = mean - reach;
      upper[d] = mean + reach;
      TORCH_CHECK(std::isfinite(lower[d]) && std::isfinite(upper[d]),
                  "Gaussian support bounds must be finite");
    }

    check_support_work(*topology, lower, upper, options.max_associations);
    topology->for_each_overlap(lower, upper, [&](int64_t voxel, double) {
      if (static_cast<int64_t>(result.gaussian.size()) >=
          options.max_associations)
        throw std::length_error(
            "Gaussian-to-voxel associations exceed max_associations");
      result.gaussian.push_back(g);
      result.voxel.push_back(voxel);
    });
  }
  return result;
}

at::Tensor long_tensor(const std::vector<int64_t> &values) {
  return at::from_blob(const_cast<int64_t *>(values.data()),
                       {static_cast<int64_t>(values.size())}, at::kLong)
      .clone();
}

} // namespace

GaussianVoxelOutput rasterize(
    const gaussian::GaussianTensor &gaussians,
    const adaptive::AdaptiveSparseTensor &reference,
    const GaussianVoxelOptions &options, const at::Tensor &confidence) {
  validate_options(options);
  validate_cpu_gaussians(gaussians);
  TORCH_CHECK(reference.device().is_cpu(),
              "the initial Gaussian-to-voxel reference implementation "
              "requires CPU reference features");
  TORCH_CHECK(reference.dimension() == 3,
              "Gaussian-to-voxel rasterization requires a 3D reference grid");

  const auto associations = find_associations(gaussians, reference, options);
  const int64_t voxel_count = reference.size();
  const int64_t channels = gaussians.feature_channels();
  const auto &positions = gaussians.positions();
  const auto options_tensor = positions.options();
  auto density = at::zeros({voxel_count, 1}, options_tensor);
  auto feature_sum = at::zeros({voxel_count, channels}, options_tensor);

  if (!associations.gaussian.empty()) {
    auto gaussian_ids = long_tensor(associations.gaussian);
    auto voxel_ids = long_tensor(associations.voxel);
    auto covariances = gaussians.covariances();
    auto precision = at::linalg_inv(covariances);
    auto determinants = at::linalg_det(covariances);
    TORCH_CHECK(at::isfinite(determinants).all().item<bool>() &&
                    (determinants > 0).all().item<bool>(),
                "Gaussian covariance matrices must be positive definite");

    const auto centers = reference.centers().slice(1, 0, 3).to(options_tensor);
    const auto voxel_volumes = reference.volumes().to(options_tensor);
    const auto selected_centers = centers.index_select(0, voxel_ids);
    const auto selected_positions = positions.index_select(0, gaussian_ids);
    const auto delta = selected_centers - selected_positions;
    const auto selected_precision = precision.index_select(0, gaussian_ids);
    const auto quadratic =
        at::bmm(at::bmm(delta.unsqueeze(1), selected_precision),
                delta.unsqueeze(2))
            .reshape({-1, 1})
            .clamp_min(0.0);

    auto amplitudes = gaussians.opacities().index_select(0, gaussian_ids) *
                     confidence_or_ones(confidence, gaussians)
                         .index_select(0, gaussian_ids) *
                     options.opacity_density_scale;
    const auto normalization =
        (std::pow(kTwoPi, 1.5) *
         determinants.index_select(0, gaussian_ids).sqrt())
            .reciprocal()
            .unsqueeze(1);
    const auto contribution =
        amplitudes * normalization * at::exp(-0.5 * quadratic) *
        voxel_volumes.index_select(0, voxel_ids).unsqueeze(1);
    density = density.index_add(0, voxel_ids, contribution);

    if (channels > 0) {
      const auto feature_values =
          gaussians.features().index_select(0, gaussian_ids);
      const auto weighted_features = feature_values * contribution;
      feature_sum = feature_sum.index_add(0, voxel_ids, weighted_features);
      if (options.feature_aggregation ==
          FeatureAggregation::NormalizedWeightedMean) {
        const auto weights =
            at::zeros({voxel_count, 1}, options_tensor)
                .index_add(0, voxel_ids, contribution);
        const double denominator_floor =
            positions.scalar_type() == at::kFloat
                ? static_cast<double>(std::numeric_limits<float>::min())
                : std::numeric_limits<double>::min();
        feature_sum =
            feature_sum / weights.clamp_min(denominator_floor);
      }
    }
  } else {
    (void)confidence_or_ones(confidence, gaussians);
  }

  at::Tensor occupancy;
  switch (options.occupancy_model) {
  case OccupancyModel::Poisson:
    occupancy = -at::expm1(-density);
    break;
  case OccupancyModel::ClampedLinear:
    occupancy = density.clamp(0.0, 1.0);
    break;
  default:
    TORCH_CHECK(false, "unknown occupancy model");
  }

  auto voxel_features = adaptive::AdaptiveSparseTensor::from_topology(
      reference.topology(), feature_sum);
  return {std::move(voxel_features), density, occupancy};
}

} // namespace gaussian_voxel
} // namespace riemann
