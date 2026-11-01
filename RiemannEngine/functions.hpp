#ifndef RIEMANN_ENGINE_FUNCTIONS_HPP
#define RIEMANN_ENGINE_FUNCTIONS_HPP

#include "engine/tensor/sparse_tensor.hpp"

#include <torch/torch.h>

namespace riemann {
namespace functional {

inline SparseTensor wrap(const SparseTensor& input, at::Tensor features) {
  return SparseTensor(std::move(features), input.coordinate_map_key(),
                      input.coordinate_manager());
}
inline SparseTensor wrap_tensor(const SparseTensor& input,
                                at::Tensor features) {
  return wrap(input, std::move(features));
}

inline SparseTensor relu(const SparseTensor& x, bool inplace = false) {
  auto y = x.F();
  if (inplace) at::relu_(y); else y = at::relu(y);
  return wrap(x, std::move(y));
}
inline SparseTensor leaky_relu(const SparseTensor& x, double slope = 0.01,
                               bool inplace = false) {
  auto y = x.F();
  if (inplace) at::leaky_relu_(y, slope); else y = at::leaky_relu(y, slope);
  return wrap(x, std::move(y));
}
inline SparseTensor elu(const SparseTensor& x, double alpha = 1.0,
                        bool inplace = false) {
  auto y = x.F();
  if (inplace) at::elu_(y, alpha, 1.0, 0.0); else y = at::elu(y, alpha, 1.0, 0.0);
  return wrap(x, std::move(y));
}
inline SparseTensor selu(const SparseTensor& x, bool inplace = false) {
  auto y = x.F();
  if (inplace) at::selu_(y); else y = at::selu(y);
  return wrap(x, std::move(y));
}
inline SparseTensor celu(const SparseTensor& x, double alpha = 1.0,
                         bool inplace = false) {
  auto y = x.F();
  if (inplace) at::celu_(y, alpha); else y = at::celu(y, alpha);
  return wrap(x, std::move(y));
}
inline SparseTensor gelu(const SparseTensor& x) { return wrap(x, at::gelu(x.F())); }
inline SparseTensor sigmoid(const SparseTensor& x) { return wrap(x, at::sigmoid(x.F())); }
inline SparseTensor tanh(const SparseTensor& x) { return wrap(x, at::tanh(x.F())); }
inline SparseTensor hardtanh(const SparseTensor& x, double min = -1.0,
                             double max = 1.0, bool inplace = false) {
  auto y = x.F();
  if (inplace) at::hardtanh_(y, min, max); else y = at::hardtanh(y, min, max);
  return wrap(x, std::move(y));
}
inline SparseTensor hardswish(const SparseTensor& x) { return wrap(x, at::hardswish(x.F())); }
inline SparseTensor relu6(const SparseTensor& x, bool inplace = false) {
  return hardtanh(x, 0.0, 6.0, inplace);
}
inline SparseTensor hardsigmoid(const SparseTensor& x) { return wrap(x, at::hardsigmoid(x.F())); }
inline SparseTensor silu(const SparseTensor& x) { return wrap(x, at::silu(x.F())); }
inline SparseTensor logsigmoid(const SparseTensor& x) { return wrap(x, at::log_sigmoid(x.F())); }
inline SparseTensor softplus(const SparseTensor& x, double beta = 1.0,
                             double threshold = 20.0) {
  return wrap(x, at::softplus(x.F(), beta, threshold));
}
inline SparseTensor softsign(const SparseTensor& x) {
  return wrap(x, torch::nn::functional::softsign(x.F()));
}
inline SparseTensor softshrink(const SparseTensor& x, double lambda = 0.5) {
  return wrap(x, at::softshrink(x.F(), lambda));
}
inline SparseTensor hardshrink(const SparseTensor& x, double lambda = 0.5) {
  return wrap(x, at::hardshrink(x.F(), lambda));
}
inline SparseTensor tanhshrink(const SparseTensor& x) {
  return wrap(x, torch::nn::functional::tanhshrink(x.F()));
}
inline SparseTensor softmax(const SparseTensor& x, int64_t dim = -1) {
  return wrap(x, at::softmax(x.F(), dim));
}
inline SparseTensor softmin(const SparseTensor& x, int64_t dim = -1) {
  return wrap(x, at::softmax(-x.F(), dim));
}
inline SparseTensor log_softmax(const SparseTensor& x, int64_t dim = -1) {
  return wrap(x, at::log_softmax(x.F(), dim));
}
inline SparseTensor glu(const SparseTensor& x, int64_t dim = -1) {
  return wrap(x, at::glu(x.F(), dim));
}
inline SparseTensor prelu(const SparseTensor& x, const at::Tensor& weight) {
  return wrap(x, at::prelu(x.F(), weight));
}
inline SparseTensor rrelu(const SparseTensor& x, double lower = 1.0 / 8.0,
                          double upper = 1.0 / 3.0, bool training = false) {
  return wrap(x, torch::nn::functional::rrelu(
                    x.F(), torch::nn::functional::RReLUFuncOptions()
                               .lower(lower).upper(upper).training(training)));
}
inline SparseTensor threshold(const SparseTensor& x, double threshold_value,
                              double value, bool inplace = false) {
  auto y = x.F();
  if (inplace) at::threshold_(y, threshold_value, value);
  else y = at::threshold(y, threshold_value, value);
  return wrap(x, std::move(y));
}
inline SparseTensor linear(const SparseTensor& x, const at::Tensor& weight,
                           const c10::optional<at::Tensor>& bias = c10::nullopt) {
  return wrap(x, at::linear(x.F(), weight, bias.has_value() ? *bias : at::Tensor()));
}
inline SparseTensor dropout(const SparseTensor& x, double p = 0.5,
                            bool training = true, bool inplace = false) {
  auto y = x.F();
  if (inplace) at::dropout_(y, p, training); else y = at::dropout(y, p, training);
  return wrap(x, std::move(y));
}
inline SparseTensor alpha_dropout(const SparseTensor& x, double p = 0.5,
                                 bool training = false, bool inplace = false) {
  auto y = x.F();
  if (inplace) at::alpha_dropout_(y, p, training);
  else y = at::alpha_dropout(y, p, training);
  return wrap(x, std::move(y));
}
inline SparseTensor normalize(const SparseTensor& x, double p = 2.0,
                              int64_t dim = 1, double eps = 1e-12) {
  return wrap(x, torch::nn::functional::normalize(
                    x.F(), torch::nn::functional::NormalizeFuncOptions()
                               .p(p).dim(dim).eps(eps)));
}

inline SparseTensor batch_norm(
    const SparseTensor& x, const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const c10::optional<at::Tensor>& weight = c10::nullopt,
    const c10::optional<at::Tensor>& bias = c10::nullopt,
    bool training = false, double momentum = 0.1, double eps = 1e-5) {
  auto options = torch::nn::functional::BatchNormFuncOptions()
                     .training(training).momentum(momentum).eps(eps);
  if (weight.has_value()) options.weight(*weight);
  if (bias.has_value()) options.bias(*bias);
  return wrap(x, torch::nn::functional::batch_norm(
                     x.F(), running_mean, running_var, options));
}
inline at::Tensor binary_cross_entropy(
    const SparseTensor& x, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight = c10::nullopt) {
  auto options = torch::nn::functional::BinaryCrossEntropyFuncOptions();
  if (weight.has_value()) options.weight(*weight);
  return torch::nn::functional::binary_cross_entropy(x.F(), target, options);
}
inline at::Tensor binary_cross_entropy_with_logits(
    const SparseTensor& x, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight = c10::nullopt) {
  auto options = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions();
  if (weight.has_value()) options.weight(*weight);
  return torch::nn::functional::binary_cross_entropy_with_logits(
      x.F(), target, options);
}
inline at::Tensor cross_entropy(const SparseTensor& x, const at::Tensor& target) {
  return torch::nn::functional::cross_entropy(x.F(), target);
}
inline at::Tensor mse_loss(const SparseTensor& x, const at::Tensor& target) {
  return at::mse_loss(x.F(), target);
}
inline at::Tensor l1_loss(const SparseTensor& x, const at::Tensor& target) {
  return at::l1_loss(x.F(), target);
}
inline at::Tensor smooth_l1_loss(const SparseTensor& x, const at::Tensor& target) {
  return at::smooth_l1_loss(x.F(), target);
}
inline at::Tensor nll_loss(const SparseTensor& x, const at::Tensor& target) {
  return at::nll_loss(x.F(), target);
}
inline at::Tensor kl_div(const SparseTensor& x, const at::Tensor& target) {
  return torch::nn::functional::kl_div(x.F(), target);
}
inline at::Tensor hinge_embedding_loss(const SparseTensor& x,
                                       const at::Tensor& target) {
  return torch::nn::functional::hinge_embedding_loss(x.F(), target);
}
inline at::Tensor poisson_nll_loss(const SparseTensor& x,
                                   const at::Tensor& target,
                                   bool log_input = true, bool full = false,
                                   double eps = 1e-8) {
  return torch::nn::functional::poisson_nll_loss(
      x.F(), target,
      torch::nn::functional::PoissonNLLLossFuncOptions()
          .log_input(log_input).full(full).eps(eps));
}
inline at::Tensor multi_margin_loss(const SparseTensor& x,
                                    const at::Tensor& target) {
  return torch::nn::functional::multi_margin_loss(x.F(), target);
}
inline at::Tensor multilabel_margin_loss(const SparseTensor& x,
                                         const at::Tensor& target) {
  return torch::nn::functional::multilabel_margin_loss(x.F(), target);
}
inline at::Tensor multilabel_soft_margin_loss(const SparseTensor& x,
                                              const at::Tensor& target) {
  return torch::nn::functional::multilabel_soft_margin_loss(x.F(), target);
}
inline at::Tensor soft_margin_loss(const SparseTensor& x, const at::Tensor& target) {
  return at::soft_margin_loss(x.F(), target);
}

} // namespace functional
} // namespace riemann

#endif
