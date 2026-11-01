#include "engine/nn/normalization.hpp"

namespace riemann {
namespace {

at::Tensor instance_normalize(const SparseTensor& input,
                              const at::Tensor& weight,
                              const at::Tensor& bias, int64_t channels,
                              double eps) {
  TORCH_CHECK(input.F().size(1) == channels,
              "Feature dimension mismatch: expected ", channels, " got ",
              input.F().size(1));
  const auto coordinates = input.C();
  const auto rows = input.F().size(0);
  if (rows == 0)
    return input.F() * weight + bias;

  const auto batch_ids = coordinates.select(1, 0)
                             .to(input.F().device(), torch::kLong);
  const auto batch_count = coordinates.select(1, 0).max().item<int64_t>() + 1;
  auto output = torch::zeros_like(input.F());
  for (int64_t batch = 0; batch < batch_count; ++batch) {
    auto indexes = at::nonzero(batch_ids == batch).reshape({-1});
    if (indexes.numel() == 0) continue;
    auto values = input.F().index_select(0, indexes);
    auto centered = values - values.mean(0, true);
    auto normalized =
        centered * torch::rsqrt(centered.square().mean(0, true) + eps);
    output = output.index_copy(0, indexes, normalized);
  }
  return output * weight + bias;
}

void reset_affine(torch::Tensor& weight, torch::Tensor& bias) {
  torch::NoGradGuard guard;
  weight.fill_(1);
  bias.zero_();
}

} // namespace

RiemannInstanceNormImpl::RiemannInstanceNormImpl(int64_t num_features)
    : num_features_(num_features) {
  TORCH_CHECK(num_features > 0, "num_features must be positive");
  weight_ = register_parameter("weight", torch::empty({1, num_features}));
  bias_ = register_parameter("bias", torch::empty({1, num_features}));
  reset();
}

void RiemannInstanceNormImpl::reset() {
  if (!weight_.defined()) return;
  reset_affine(weight_, bias_);
}

SparseTensor RiemannInstanceNormImpl::forward(const SparseTensor& input) {
  auto output = instance_normalize(input, weight_, bias_, num_features_, eps_);
  return SparseTensor(std::move(output), input.coordinate_map_key(),
                      input.coordinate_manager());
}

RiemannStableInstanceNormImpl::RiemannStableInstanceNormImpl(
    int64_t num_features)
    : num_features_(num_features) {
  TORCH_CHECK(num_features > 0, "num_features must be positive");
  weight_ = register_parameter("weight", torch::empty({1, num_features}));
  bias_ = register_parameter("bias", torch::empty({1, num_features}));
  reset();
}

void RiemannStableInstanceNormImpl::reset() {
  if (!weight_.defined()) return;
  reset_affine(weight_, bias_);
}

SparseTensor
RiemannStableInstanceNormImpl::forward(const SparseTensor& input) {
  auto output = instance_normalize(input, weight_, bias_, num_features_, eps_);
  return SparseTensor(std::move(output), input.coordinate_map_key(),
                      input.coordinate_manager());
}

} // namespace riemann
