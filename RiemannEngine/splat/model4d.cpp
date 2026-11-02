#include "splat/model4d.hpp"

#include "splat/harmonics.hpp"

#include <cmath>
#include <utility>

namespace riemann {
namespace gaussian {

Gaussian4DModel::Gaussian4DModel(at::Tensor xyz, at::Tensor f_dc,
                                 at::Tensor f_rest, at::Tensor opacity_raw,
                                 at::Tensor scale_raw, at::Tensor rotation_raw,
                                 at::Tensor velocity, at::Tensor time_center,
                                 at::Tensor time_scale_raw)
    : xyz_(std::move(xyz)), f_dc_(std::move(f_dc)), f_rest_(std::move(f_rest)),
      opacity_raw_(std::move(opacity_raw)), scale_raw_(std::move(scale_raw)),
      rotation_raw_(std::move(rotation_raw)), velocity_(std::move(velocity)),
      time_center_(std::move(time_center)),
      time_scale_raw_(std::move(time_scale_raw)) {
  check_invariants();
}

void Gaussian4DModel::check_invariants() const {
  // The spatial parameters share GaussianModel's invariants.
  GaussianModel spatial(xyz_, f_dc_, f_rest_, opacity_raw_, scale_raw_,
                        rotation_raw_);
  const auto rows = xyz_.size(0);
  TORCH_CHECK(velocity_.dim() == 2 && velocity_.size(0) == rows &&
                  velocity_.size(1) == 3,
              "velocity must have shape [N, 3]");
  TORCH_CHECK(time_center_.dim() == 2 && time_center_.size(0) == rows &&
                  time_center_.size(1) == 1,
              "time_center must have shape [N, 1]");
  TORCH_CHECK(time_scale_raw_.dim() == 2 && time_scale_raw_.size(0) == rows &&
                  time_scale_raw_.size(1) == 1,
              "time_scale_raw must have shape [N, 1]");
  for (const auto *parameter : {&velocity_, &time_center_, &time_scale_raw_}) {
    TORCH_CHECK(parameter->scalar_type() == at::kFloat,
                "Gaussian model parameters must be float32");
    TORCH_CHECK(parameter->device() == xyz_.device(),
                "Gaussian model parameters must share a device");
  }
}

Gaussian4DModel Gaussian4DModel::from_model(const GaussianModel &model,
                                            double initial_time_scale) {
  TORCH_CHECK(std::isfinite(initial_time_scale) && initial_time_scale > 0.0,
              "initial_time_scale must be finite and positive");
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(model.device());
  const auto rows = model.size();
  return Gaussian4DModel(
      model.xyz().detach().clone(), model.f_dc().detach().clone(),
      model.f_rest().detach().clone(), model.opacity_raw().detach().clone(),
      model.scale_raw().detach().clone(),
      model.rotation_raw().detach().clone(), at::zeros({rows, 3}, options),
      at::full({rows, 1}, 0.5, options),
      at::full({rows, 1}, std::log(initial_time_scale), options));
}

Gaussian4DModel Gaussian4DModel::from_points(const at::Tensor &positions,
                                             const at::Tensor &colors,
                                             int sh_degree,
                                             double initial_opacity,
                                             double initial_time_scale) {
  TORCH_CHECK(std::isfinite(initial_time_scale) && initial_time_scale > 0.0,
              "initial_time_scale must be finite and positive");
  const auto spatial =
      GaussianModel::from_points(positions, colors, sh_degree,
                                 initial_opacity);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(spatial.device());
  const auto rows = spatial.size();
  return Gaussian4DModel(
      spatial.xyz(), spatial.f_dc(), spatial.f_rest(), spatial.opacity_raw(),
      spatial.scale_raw(), spatial.rotation_raw(),
      at::zeros({rows, 3}, options), at::rand({rows, 1}, options),
      at::full({rows, 1}, std::log(initial_time_scale), options));
}

int Gaussian4DModel::sh_degree() const {
  switch (f_rest_.size(1)) {
  case 0:
    return 0;
  case 3:
    return 1;
  case 8:
    return 2;
  default:
    return 3;
  }
}

at::Tensor Gaussian4DModel::rotations() const {
  return rotation_raw_ / rotation_raw_.norm(2, {1}, true).clamp_min(1e-12);
}

at::Tensor Gaussian4DModel::scales() const { return at::exp(scale_raw_); }

at::Tensor Gaussian4DModel::time_scales() const {
  return at::exp(time_scale_raw_);
}

at::Tensor Gaussian4DModel::opacities() const {
  return at::sigmoid(opacity_raw_);
}

at::Tensor Gaussian4DModel::sh_coefficients() const {
  return at::cat({f_dc_.unsqueeze(1), f_rest_}, 1);
}

at::Tensor Gaussian4DModel::positions(double time) const {
  TORCH_CHECK(std::isfinite(time), "time must be finite");
  return xyz_ + velocity_ * (time - time_center_);
}

at::Tensor Gaussian4DModel::opacities(double time) const {
  TORCH_CHECK(std::isfinite(time), "time must be finite");
  const auto normalized =
      (time - time_center_) / time_scales().clamp_min(1e-6);
  return opacities() * at::exp(-0.5 * normalized * normalized);
}

GaussianModel Gaussian4DModel::at_time(double time) const {
  torch::NoGradGuard no_grad;
  constexpr double kLogitEpsilon = 1e-6;
  const auto opacity =
      opacities(time).clamp(kLogitEpsilon, 1.0 - kLogitEpsilon);
  return GaussianModel(positions(time).detach().clone(),
                       f_dc_.detach().clone(), f_rest_.detach().clone(),
                       at::log(opacity / (1.0 - opacity)).detach().clone(),
                       scale_raw_.detach().clone(),
                       rotation_raw_.detach().clone());
}

std::vector<at::Tensor> Gaussian4DModel::parameters() const {
  return {xyz_,          f_dc_,     f_rest_,
          opacity_raw_,  scale_raw_, rotation_raw_,
          velocity_,     time_center_, time_scale_raw_};
}

void Gaussian4DModel::set_parameters(
    const std::vector<at::Tensor> &parameters) {
  TORCH_CHECK(parameters.size() == 9,
              "expected 9 4D Gaussian parameter tensors");
  Gaussian4DModel replacement(parameters[0], parameters[1], parameters[2],
                              parameters[3], parameters[4], parameters[5],
                              parameters[6], parameters[7], parameters[8]);
  *this = std::move(replacement);
}

std::vector<at::Tensor> Gaussian4DModel::enable_gradients() {
  for (auto *parameter :
       {&xyz_, &f_dc_, &f_rest_, &opacity_raw_, &scale_raw_, &rotation_raw_,
        &velocity_, &time_center_, &time_scale_raw_})
    *parameter = parameter->detach().set_requires_grad(true);
  return parameters();
}

Gaussian4DModel Gaussian4DModel::to(torch::Device target) const {
  auto moved = [&](const at::Tensor &t) { return t.detach().to(target); };
  return Gaussian4DModel(moved(xyz_), moved(f_dc_), moved(f_rest_),
                         moved(opacity_raw_), moved(scale_raw_),
                         moved(rotation_raw_), moved(velocity_),
                         moved(time_center_), moved(time_scale_raw_));
}

Gaussian4DModel Gaussian4DModel::clone() const {
  auto copied = [](const at::Tensor &t) { return t.detach().clone(); };
  return Gaussian4DModel(copied(xyz_), copied(f_dc_), copied(f_rest_),
                         copied(opacity_raw_), copied(scale_raw_),
                         copied(rotation_raw_), copied(velocity_),
                         copied(time_center_), copied(time_scale_raw_));
}

Gaussian4DModel Gaussian4DModel::select(const at::Tensor &indices) const {
  TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == at::kLong,
              "indices must be a one-dimensional int64 tensor");
  auto picked = [&](const at::Tensor &t) {
    return t.detach().index_select(0, indices);
  };
  return Gaussian4DModel(picked(xyz_), picked(f_dc_), picked(f_rest_),
                         picked(opacity_raw_), picked(scale_raw_),
                         picked(rotation_raw_), picked(velocity_),
                         picked(time_center_), picked(time_scale_raw_));
}

} // namespace gaussian
} // namespace riemann
