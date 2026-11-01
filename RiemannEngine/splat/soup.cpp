#include "splat/soup.hpp"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <utility>

#ifdef USE_CUDA
#include "splat/gpu.hpp"
#endif

namespace riemann {
namespace gaussian {
namespace {

void require_field(const at::Tensor &field, int64_t rows, int64_t columns,
                   const char *name) {
  TORCH_CHECK(field.defined(), name, " must be defined");
  TORCH_CHECK(field.dim() == 2 && field.size(1) == columns, name,
              " must have shape [N, ", columns, "]");
  TORCH_CHECK(field.size(0) == rows, name, " row count must match positions");
}

at::Tensor unit_quaternions(const at::Tensor &quaternions) {
  const auto norms = quaternions.square().sum(1, true).sqrt();
  const auto usable = at::isfinite(quaternions).all() & (norms > 0).all() &
                      at::isfinite(norms).all();
  TORCH_CHECK(usable.item<bool>(),
              "quaternion rotations must have finite nonzero norm");
  return quaternions / norms;
}

at::Tensor quaternion_matrices(const at::Tensor &quaternions) {
  const auto u = unit_quaternions(quaternions);
  const auto w = u.select(1, 0);
  const auto x = u.select(1, 1);
  const auto y = u.select(1, 2);
  const auto z = u.select(1, 3);
  const auto two = at::full_like(w, 2);
  return at::stack({1 - two * (y * y + z * z), two * (x * y - z * w),
                    two * (x * z + y * w), two * (x * y + z * w),
                    1 - two * (x * x + z * z), two * (y * z - x * w),
                    two * (x * z - y * w), two * (y * z + x * w),
                    1 - two * (x * x + y * y)},
                   1)
      .reshape({quaternions.size(0), 3, 3});
}

at::Tensor compose_quaternions(const at::Tensor &lhs, const at::Tensor &rhs) {
  const auto pw = lhs.select(1, 0), px = lhs.select(1, 1);
  const auto py = lhs.select(1, 2), pz = lhs.select(1, 3);
  const auto qw = rhs.select(1, 0), qx = rhs.select(1, 1);
  const auto qy = rhs.select(1, 2), qz = rhs.select(1, 3);
  return at::stack({pw * qw - px * qx - py * qy - pz * qz,
                    pw * qx + px * qw + py * qz - pz * qy,
                    pw * qy - px * qz + py * qw + pz * qx,
                    pw * qz + px * qy - py * qx + pz * qw},
                   1);
}

at::Tensor rigidly_move(const at::Tensor &points, const at::Tensor &rotation,
                        const at::Tensor &translation) {
  const auto matrices = quaternion_matrices(rotation);
  return at::bmm(matrices, points.unsqueeze(2)).squeeze(2) + translation;
}

} // namespace

GaussianTensor::GaussianTensor(at::Tensor positions, at::Tensor rotations,
                               at::Tensor scales, at::Tensor opacities,
                               at::Tensor colors, at::Tensor features,
                               at::Tensor velocities)
    : positions_(std::move(positions)), rotations_(std::move(rotations)),
      scales_(std::move(scales)), opacities_(std::move(opacities)),
      colors_(std::move(colors)), features_(std::move(features)),
      velocities_(std::move(velocities)) {
  TORCH_CHECK(positions_.defined() && positions_.dim() == 2 &&
                  positions_.size(1) == 3,
              "positions must have shape [N, 3]");
  if (!features_.defined())
    features_ = positions_.new_empty({positions_.size(0), 0});
  check_invariants();
}

GaussianTensor GaussianTensor::empty(int64_t count,
                                     const at::TensorOptions &options,
                                     int64_t feature_channels,
                                     bool with_velocities) {
  TORCH_CHECK(count >= 0, "Gaussian count must be nonnegative");
  TORCH_CHECK(feature_channels >= 0,
              "feature channel count must be nonnegative");
  return GaussianTensor(
      at::empty({count, 3}, options), at::empty({count, 4}, options),
      at::empty({count, 3}, options), at::empty({count, 1}, options),
      at::empty({count, 3}, options),
      at::empty({count, feature_channels}, options),
      with_velocities ? at::empty({count, 3}, options) : at::Tensor());
}

void GaussianTensor::check_invariants() const {
  TORCH_CHECK(positions_.defined() && positions_.dim() == 2 &&
                  positions_.size(1) == 3,
              "positions must have shape [N, 3]");
  const auto rows = positions_.size(0);
  require_field(rotations_, rows, 4, "rotations");
  require_field(scales_, rows, 3, "scales");
  require_field(opacities_, rows, 1, "opacities");
  require_field(colors_, rows, 3, "colors");
  require_field(features_, rows,
                features_.dim() == 2 ? features_.size(1) : -1, "features");
  if (velocities_.defined())
    require_field(velocities_, rows, 3, "velocities");

  TORCH_CHECK(positions_.is_floating_point() &&
                  (positions_.scalar_type() == at::kFloat ||
                   positions_.scalar_type() == at::kDouble),
              "Gaussian tensors support float32 and float64");
  const auto shared_device = positions_.device();
  const auto shared_type = positions_.scalar_type();
  for (const auto *field :
       {&rotations_, &scales_, &opacities_, &colors_, &features_}) {
    TORCH_CHECK(field->device() == shared_device,
                "all Gaussian fields must share a device");
    TORCH_CHECK(field->scalar_type() == shared_type,
                "all Gaussian fields must share a dtype");
    TORCH_CHECK(field->is_floating_point(),
                "Gaussian fields must use a floating-point dtype");
  }
  if (velocities_.defined()) {
    TORCH_CHECK(velocities_.device() == shared_device,
                "all Gaussian fields must share a device");
    TORCH_CHECK(velocities_.scalar_type() == shared_type,
                "all Gaussian fields must share a dtype");
  }
}

at::Tensor GaussianTensor::covariances() const {
#ifdef USE_CUDA
  if (positions_.is_cuda() && !rotations_.requires_grad() &&
      !scales_.requires_grad()) {
    const auto rotations = unit_quaternions(rotations_.contiguous());
    const auto scales = scales_.contiguous();
    auto output = at::empty({size(), 3, 3}, positions_.options());
    gpu::covariance_cuda(rotations, scales, output);
    return output;
  }
#endif
  const auto rotation = quaternion_matrices(rotations_);
  const auto variances = at::diag_embed(scales_.square());
  return at::bmm(at::bmm(rotation, variances), rotation.transpose(1, 2));
}

GaussianTensor GaussianTensor::select(const at::Tensor &indices) const {
  TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == at::kLong,
              "indices must be a one-dimensional int64 tensor");
  TORCH_CHECK(indices.device() == device(),
              "indices and Gaussian tensors must be on the same device");
  return GaussianTensor(
      positions_.index_select(0, indices), rotations_.index_select(0, indices),
      scales_.index_select(0, indices), opacities_.index_select(0, indices),
      colors_.index_select(0, indices), features_.index_select(0, indices),
      velocities_.defined() ? velocities_.index_select(0, indices)
                            : at::Tensor());
}

GaussianTensor GaussianTensor::remove(const at::Tensor &indices) const {
  TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == at::kLong,
              "indices must be a one-dimensional int64 tensor");
  TORCH_CHECK(indices.device() == device(),
              "indices and Gaussian tensors must be on the same device");
  auto keep = at::ones({size()}, positions_.options().dtype(at::kBool));
  keep.index_fill_(0, indices, false);
  return select(at::nonzero(keep).reshape({-1}));
}

GaussianTensor GaussianTensor::filter_opacity(double minimum_opacity) const {
  TORCH_CHECK(std::isfinite(minimum_opacity) && minimum_opacity >= 0.0 &&
                  minimum_opacity <= 1.0,
              "minimum opacity must be finite and in [0, 1]");
  const auto kept =
      at::nonzero(opacities_.select(1, 0) >= minimum_opacity).reshape({-1});
  return select(kept);
}

GaussianTensor GaussianTensor::filter_bounds(const at::Tensor &lower,
                                             const at::Tensor &upper) const {
  TORCH_CHECK(lower.sizes() == at::IntArrayRef({3}) &&
                  upper.sizes() == at::IntArrayRef({3}),
              "lower and upper bounds must have shape [3]");
  TORCH_CHECK(lower.device() == device() && upper.device() == device(),
              "bounds and Gaussian tensors must be on the same device");
  TORCH_CHECK(lower.scalar_type() == dtype() && upper.scalar_type() == dtype(),
              "bounds and Gaussian tensors must have the same dtype");
  TORCH_CHECK(at::all(lower < upper).item<bool>(),
              "each lower bound must be smaller than its upper bound");
  const auto inside = ((positions_ >= lower) & (positions_ < upper)).all(1);
  return select(at::nonzero(inside).reshape({-1}));
}

GaussianTensor GaussianTensor::append(const GaussianTensor &other) const {
  return concatenate({*this, other});
}

GaussianTensor
GaussianTensor::concatenate(const std::vector<GaussianTensor> &tensors) {
  TORCH_CHECK(!tensors.empty(),
              "cannot concatenate an empty Gaussian tensor list");
  const auto &head = tensors.front();
  const bool with_velocities = head.has_velocities();
  std::vector<at::Tensor> positions, rotations, scales, opacities, colors,
      features, velocities;
  positions.reserve(tensors.size());
  rotations.reserve(tensors.size());
  scales.reserve(tensors.size());
  opacities.reserve(tensors.size());
  colors.reserve(tensors.size());
  features.reserve(tensors.size());
  for (const auto &entry : tensors) {
    TORCH_CHECK(entry.device() == head.device() &&
                    entry.dtype() == head.dtype() &&
                    entry.feature_channels() == head.feature_channels() &&
                    entry.has_velocities() == with_velocities,
                "concatenated Gaussian tensors must have matching device, "
                "dtype, feature channels, and velocity presence");
    positions.push_back(entry.positions_);
    rotations.push_back(entry.rotations_);
    scales.push_back(entry.scales_);
    opacities.push_back(entry.opacities_);
    colors.push_back(entry.colors_);
    features.push_back(entry.features_);
    if (with_velocities)
      velocities.push_back(entry.velocities_);
  }
  return GaussianTensor(at::cat(positions, 0), at::cat(rotations, 0),
                        at::cat(scales, 0), at::cat(opacities, 0),
                        at::cat(colors, 0), at::cat(features, 0),
                        with_velocities ? at::cat(velocities, 0)
                                        : at::Tensor());
}

GaussianTensor GaussianTensor::clone() const {
  return GaussianTensor(positions_.clone(), rotations_.clone(),
                        scales_.clone(), opacities_.clone(), colors_.clone(),
                        features_.clone(),
                        velocities_.defined() ? velocities_.clone()
                                              : at::Tensor());
}

GaussianTensor GaussianTensor::to(torch::Device target) const {
  return GaussianTensor(positions_.to(target), rotations_.to(target),
                        scales_.to(target), opacities_.to(target),
                        colors_.to(target), features_.to(target),
                        velocities_.defined() ? velocities_.to(target)
                                              : at::Tensor());
}

std::vector<at::Tensor> GaussianTensor::enable_gradients() {
  positions_ = positions_.detach().set_requires_grad(true);
  rotations_ = rotations_.detach().set_requires_grad(true);
  scales_ = scales_.detach().set_requires_grad(true);
  opacities_ = opacities_.detach().set_requires_grad(true);
  colors_ = colors_.detach().set_requires_grad(true);
  std::vector<at::Tensor> parameters{positions_, rotations_, scales_,
                                     opacities_, colors_};
  if (features_.numel() > 0) {
    features_ = features_.detach().set_requires_grad(true);
    parameters.push_back(features_);
  }
  if (velocities_.defined()) {
    velocities_ = velocities_.detach().set_requires_grad(true);
    parameters.push_back(velocities_);
  }
  return parameters;
}

void GaussianTensor::project_constraints() {
  at::NoGradGuard guard;
  scales_.clamp_(1e-6);
  opacities_.clamp_(0.0, 1.0);
}

GaussianTensor GaussianTensor::transform(const at::Tensor &rotation,
                                         const at::Tensor &translation) const {
  TORCH_CHECK(rotation.dim() == 1 || rotation.dim() == 2,
              "rotation must have shape [4] or [N, 4]");
  TORCH_CHECK(translation.dim() == 1 || translation.dim() == 2,
              "translation must have shape [3] or [N, 3]");
  TORCH_CHECK(rotation.size(rotation.dim() - 1) == 4 &&
                  translation.size(translation.dim() - 1) == 3,
              "rotation and translation must have final dimensions 4 and 3; "
              "got ",
              rotation.sizes(), " and ", translation.sizes());
  TORCH_CHECK(rotation.device() == device() &&
                  translation.device() == device(),
              "transform and Gaussian tensors must be on the same device");
  TORCH_CHECK(rotation.scalar_type() == dtype() &&
                  translation.scalar_type() == dtype(),
              "transform and Gaussian tensors must have the same dtype");

  const auto rotations = rotation.dim() == 1
                             ? rotation.unsqueeze(0).expand({size(), 4})
                             : rotation;
  const auto translations =
      translation.dim() == 1 ? translation.unsqueeze(0).expand({size(), 3})
                             : translation;
  TORCH_CHECK(rotations.size(0) == size() && translations.size(0) == size(),
              "batched transform row counts must equal Gaussian count");

#ifdef USE_CUDA
  if (positions_.is_cuda() && !positions_.requires_grad() &&
      !rotations_.requires_grad() && !rotation.requires_grad() &&
      !translation.requires_grad() &&
      (!velocities_.defined() || !velocities_.requires_grad())) {
    const auto rig = unit_quaternions(rotations.contiguous()).contiguous();
    const auto own = unit_quaternions(rotations_.contiguous()).contiguous();
    const auto in_positions = positions_.contiguous();
    const auto in_translations = translations.contiguous();
    const auto in_velocities =
        velocities_.defined() ? velocities_.contiguous() : at::Tensor();
    auto out_positions = at::empty_like(in_positions);
    auto out_rotations = at::empty_like(own);
    auto out_velocities =
        velocities_.defined() ? at::empty_like(in_velocities) : at::Tensor();
    gpu::transform_cuda(in_positions, own, rig, in_translations, in_velocities,
                        out_positions, out_rotations, out_velocities);
    return GaussianTensor(std::move(out_positions), std::move(out_rotations),
                          scales_, opacities_, colors_, features_,
                          std::move(out_velocities));
  }
#endif

  auto moved_positions = rigidly_move(positions_, rotations, translations);
  const auto unit_rig = unit_quaternions(rotations);
  const auto unit_own = unit_quaternions(rotations_);
  auto composed = compose_quaternions(unit_rig, unit_own);
  auto moved_velocities =
      velocities_.defined()
          ? at::bmm(quaternion_matrices(rotations), velocities_.unsqueeze(2))
                .squeeze(2)
          : at::Tensor();
  return GaussianTensor(std::move(moved_positions), std::move(composed),
                        scales_, opacities_, colors_, features_,
                        std::move(moved_velocities));
}

} // namespace gaussian
} // namespace riemann
