#ifndef RIEMANN_SPLAT_MODEL4D_HPP
#define RIEMANN_SPLAT_MODEL4D_HPP

#include "splat/model.hpp"

#include <torch/torch.h>

#include <vector>

namespace riemann {
namespace gaussian {

// Trainable 4D (space + time) Gaussian-splat parameters for dynamic scenes.
// Each Gaussian is a 4D Gaussian over (x, y, z, t) stored in conditional
// form: evaluating the scene at a time t yields a 3D Gaussian whose mean
// moves linearly, xyz + velocity * (t - time_center), and whose opacity is
// modulated by the marginal temporal Gaussian,
// exp(-0.5 * ((t - time_center) / time_scale)^2).
//
// Raw parameters extend GaussianModel:
//   xyz [N,3], f_dc [N,3], f_rest [N,K-1,3], opacity_raw [N,1] (logit),
//   scale_raw [N,3] (log), rotation_raw [N,4] (unnormalized w x y z),
//   velocity [N,3] (units of position per unit time),
//   time_center [N,1], time_scale_raw [N,1] (log of the temporal std).
class Gaussian4DModel {
public:
  Gaussian4DModel(at::Tensor xyz, at::Tensor f_dc, at::Tensor f_rest,
                  at::Tensor opacity_raw, at::Tensor scale_raw,
                  at::Tensor rotation_raw, at::Tensor velocity,
                  at::Tensor time_center, at::Tensor time_scale_raw);

  // 3D model lifted to 4D: zero velocity and time_center 0.5. Opacity still
  // follows the temporal Gaussian (use a large std for a near-static scene).
  static Gaussian4DModel from_model(const GaussianModel &model,
                                    double initial_time_scale = 1.0);

  // Initializes from a colored point cloud (see GaussianModel::from_points)
  // with time centers spread uniformly over [0, 1].
  static Gaussian4DModel from_points(const at::Tensor &positions,
                                     const at::Tensor &colors,
                                     int sh_degree = 3,
                                     double initial_opacity = 0.1,
                                     double initial_time_scale = 0.5);

  int64_t size() const { return xyz_.size(0); }
  int sh_degree() const;
  torch::Device device() const { return xyz_.device(); }

  const at::Tensor &xyz() const { return xyz_; }
  const at::Tensor &f_dc() const { return f_dc_; }
  const at::Tensor &f_rest() const { return f_rest_; }
  const at::Tensor &opacity_raw() const { return opacity_raw_; }
  const at::Tensor &scale_raw() const { return scale_raw_; }
  const at::Tensor &rotation_raw() const { return rotation_raw_; }
  const at::Tensor &velocity() const { return velocity_; }
  const at::Tensor &time_center() const { return time_center_; }
  const at::Tensor &time_scale_raw() const { return time_scale_raw_; }

  // Activated views (differentiable).
  at::Tensor rotations() const;
  at::Tensor scales() const;
  at::Tensor time_scales() const;
  // Peak (time-independent) opacity, sigmoid(opacity_raw).
  at::Tensor opacities() const;
  // [N, K, 3] with the DC term first.
  at::Tensor sh_coefficients() const;

  // Conditional mean at time t: xyz + velocity * (t - time_center), [N, 3].
  at::Tensor positions(double time) const;
  // Peak opacity scaled by the temporal Gaussian at time t, [N, 1].
  at::Tensor opacities(double time) const;

  // Detached static 3DGS snapshot of the scene at time t (for PLY export or
  // the 3D pipeline). Temporal attenuation is folded into the opacity.
  GaussianModel at_time(double time) const;

  // Fixed order: xyz, f_dc, f_rest, opacity_raw, scale_raw, rotation_raw,
  // velocity, time_center, time_scale_raw.
  std::vector<at::Tensor> parameters() const;
  void set_parameters(const std::vector<at::Tensor> &parameters);
  // Makes every parameter a leaf that requires gradients.
  std::vector<at::Tensor> enable_gradients();

  Gaussian4DModel to(torch::Device device) const;
  Gaussian4DModel clone() const;
  Gaussian4DModel select(const at::Tensor &indices) const;

private:
  void check_invariants() const;

  at::Tensor xyz_, f_dc_, f_rest_, opacity_raw_, scale_raw_, rotation_raw_;
  at::Tensor velocity_, time_center_, time_scale_raw_;
};

} // namespace gaussian
} // namespace riemann

#endif
