#include "splat/optimizer.hpp"

#include <c10/util/Exception.h>

#include <cmath>

namespace riemann {
namespace gaussian {
namespace {

void require_comparable(const at::Tensor &prediction, const at::Tensor &target,
                        const char *name) {
  TORCH_CHECK(prediction.defined() && target.defined(), name,
              " tensors must be defined");
  TORCH_CHECK(prediction.sizes() == target.sizes(), name,
              " prediction and target shapes must match");
  TORCH_CHECK(prediction.device() == target.device() &&
                  prediction.scalar_type() == target.scalar_type(),
              name, " prediction and target device and dtype must match");
  TORCH_CHECK(prediction.is_floating_point(), name,
              " tensors must have a floating-point dtype");
}

std::vector<at::Tensor> trainable_fields(GaussianTensor &gaussians,
                                         double learning_rate) {
  TORCH_CHECK(std::isfinite(learning_rate) && learning_rate > 0.0,
              "learning_rate must be positive and finite");
  TORCH_CHECK(gaussians.size() > 0,
              "Gaussian optimizer requires a non-empty tensor");
  return gaussians.enable_gradients();
}

} // namespace

GaussianOptimizer::GaussianOptimizer(GaussianTensor &gaussians,
                                     double learning_rate)
    : gaussians_(&gaussians),
      parameters_(trainable_fields(gaussians, learning_rate)),
      optimizer_(parameters_, torch::optim::AdamOptions(learning_rate)) {
  TORCH_CHECK(!parameters_.empty(), "Gaussian optimizer requires parameters");
}

at::Tensor
GaussianOptimizer::rgb_reconstruction_loss(const at::Tensor &prediction,
                                           const at::Tensor &target) {
  require_comparable(prediction, target, "RGB reconstruction");
  TORCH_CHECK(prediction.dim() >= 1 && prediction.size(-1) == 3,
              "RGB tensors must have a final dimension of 3");
  TORCH_CHECK(prediction.numel() > 0,
              "RGB reconstruction tensors must not be empty");
  return at::mse_loss(prediction, target);
}

at::Tensor
GaussianOptimizer::depth_reconstruction_loss(const at::Tensor &prediction,
                                             const at::Tensor &target) {
  require_comparable(prediction, target, "depth reconstruction");
  TORCH_CHECK(prediction.numel() > 0,
              "depth reconstruction tensors must not be empty");
  return at::mse_loss(prediction, target);
}

at::Tensor
GaussianOptimizer::opacity_regularization(const GaussianTensor &gaussians) {
  TORCH_CHECK(gaussians.size() > 0,
              "opacity regularization requires at least one Gaussian");
  return gaussians.opacities().square().mean();
}

at::Tensor
GaussianOptimizer::scale_regularization(const GaussianTensor &gaussians) {
  TORCH_CHECK(gaussians.size() > 0,
              "scale regularization requires at least one Gaussian");
  return gaussians.scales().clamp_min(1e-8).log().square().mean();
}

at::Tensor GaussianOptimizer::objective(const GaussianTensor &gaussians,
                                        const at::Tensor &predicted_rgb,
                                        const at::Tensor &target_rgb,
                                        double opacity_weight,
                                        double scale_weight) {
  TORCH_CHECK(std::isfinite(opacity_weight) && opacity_weight >= 0.0 &&
                  std::isfinite(scale_weight) && scale_weight >= 0.0,
              "regularization weights must be finite and nonnegative");
  auto loss = rgb_reconstruction_loss(predicted_rgb, target_rgb);
  if (opacity_weight != 0.0)
    loss = loss + opacity_weight * opacity_regularization(gaussians);
  if (scale_weight != 0.0)
    loss = loss + scale_weight * scale_regularization(gaussians);
  return loss;
}

void GaussianOptimizer::step(const at::Tensor &loss) {
  TORCH_CHECK(loss.defined() && loss.numel() == 1,
              "optimizer loss must be a defined scalar tensor");
  TORCH_CHECK(loss.requires_grad(),
              "optimizer loss must participate in an autograd graph");
  TORCH_CHECK(loss.device() == parameters_.front().device(),
              "optimizer loss and parameters must be on the same device");
  optimizer_.zero_grad();
  loss.backward();
  optimizer_.step();
  gaussians_->project_constraints();
}

void GaussianOptimizer::zero_grad() { optimizer_.zero_grad(); }

} // namespace gaussian
} // namespace riemann
