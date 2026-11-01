#include "engine/nn/activation.hpp"

namespace riemann {

RiemannSinusoidalImpl::RiemannSinusoidalImpl(int in_channels,
                                             int out_channels)
    : in_channels_(in_channels), out_channels_(out_channels) {
  TORCH_CHECK(in_channels > 0 && out_channels > 0,
              "Sinusoidal activation channels must be positive");
  kernel_ =
      register_parameter("kernel", torch::empty({in_channels, out_channels}));
  bias_ = register_parameter("bias", torch::empty({1, out_channels}));
  coef_ = register_parameter("coef", torch::empty({1, out_channels}));
  reset();
}

void RiemannSinusoidalImpl::reset() {
  if (!kernel_.defined()) return;
  torch::NoGradGuard guard;
  kernel_.uniform_(0, 1);
  bias_.uniform_(0, 1);
  coef_.uniform_(0, 1);
}

SparseTensor RiemannSinusoidalImpl::forward(const SparseTensor& input) {
  TORCH_CHECK(input.F().size(1) == in_channels_,
              "Input feature count does not match sinusoidal layer");
  auto output = torch::sin(input.F().matmul(kernel_) + bias_) * coef_;
  return SparseTensor(std::move(output), input.coordinate_map_key(),
                      input.coordinate_manager());
}

} // namespace riemann
