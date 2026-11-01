#ifndef RIEMANN_ENGINE_NN_ACTIVATION_HPP
#define RIEMANN_ENGINE_NN_ACTIVATION_HPP

#include "engine/tensor/sparse_tensor.hpp"

#include <torch/torch.h>

namespace riemann {

class RiemannReLUImpl : public torch::nn::Module {
public:
  explicit RiemannReLUImpl(bool inplace = false) : inplace_(inplace) {}
  explicit RiemannReLUImpl(torch::nn::ReLUOptions options)
      : inplace_(options.inplace()) {}
  SparseTensor forward(const SparseTensor& input) {
    auto output = input.F();
    if (inplace_)
      at::relu_(output);
    else
      output = at::relu(output);
    return SparseTensor(std::move(output), input.coordinate_map_key(),
                        input.coordinate_manager());
  }

private:
  bool inplace_;
};
TORCH_MODULE(RiemannReLU);

template <typename TorchModule>
class RiemannFeatureModuleImpl : public torch::nn::Module {
public:
  template <typename... Args>
  explicit RiemannFeatureModuleImpl(Args&&... args)
      : module_(std::forward<Args>(args)...) {
    register_module("module", module_);
  }

  SparseTensor forward(const SparseTensor& input) {
    auto output = module_->forward(input.F());
    return SparseTensor(std::move(output), input.coordinate_map_key(),
                        input.coordinate_manager());
  }

private:
  TorchModule module_{nullptr};
};

template <typename TorchModule>
using RiemannNonlinearityImpl = RiemannFeatureModuleImpl<TorchModule>;

#define RIEMANN_FEATURE_ACTIVATION(name, torch_module)                          \
  using name##Impl = RiemannFeatureModuleImpl<torch_module>;                   \
  TORCH_MODULE(name)

RIEMANN_FEATURE_ACTIVATION(RiemannReLU6, torch::nn::ReLU6);
RIEMANN_FEATURE_ACTIVATION(RiemannLeakyReLU, torch::nn::LeakyReLU);
RIEMANN_FEATURE_ACTIVATION(RiemannELU, torch::nn::ELU);
RIEMANN_FEATURE_ACTIVATION(RiemannSELU, torch::nn::SELU);
RIEMANN_FEATURE_ACTIVATION(RiemannCELU, torch::nn::CELU);
RIEMANN_FEATURE_ACTIVATION(RiemannGELU, torch::nn::GELU);
RIEMANN_FEATURE_ACTIVATION(RiemannSiLU, torch::nn::SiLU);
RIEMANN_FEATURE_ACTIVATION(RiemannSigmoid, torch::nn::Sigmoid);
RIEMANN_FEATURE_ACTIVATION(RiemannTanh, torch::nn::Tanh);
RIEMANN_FEATURE_ACTIVATION(RiemannHardshrink, torch::nn::Hardshrink);
RIEMANN_FEATURE_ACTIVATION(RiemannHardtanh, torch::nn::Hardtanh);
RIEMANN_FEATURE_ACTIVATION(RiemannLogSigmoid, torch::nn::LogSigmoid);
RIEMANN_FEATURE_ACTIVATION(RiemannPReLU, torch::nn::PReLU);
RIEMANN_FEATURE_ACTIVATION(RiemannRReLU, torch::nn::RReLU);
RIEMANN_FEATURE_ACTIVATION(RiemannSoftplus, torch::nn::Softplus);
RIEMANN_FEATURE_ACTIVATION(RiemannSoftshrink, torch::nn::Softshrink);
RIEMANN_FEATURE_ACTIVATION(RiemannSoftsign, torch::nn::Softsign);
RIEMANN_FEATURE_ACTIVATION(RiemannTanhshrink, torch::nn::Tanhshrink);
RIEMANN_FEATURE_ACTIVATION(RiemannThreshold, torch::nn::Threshold);
RIEMANN_FEATURE_ACTIVATION(RiemannSoftmin, torch::nn::Softmin);
RIEMANN_FEATURE_ACTIVATION(RiemannSoftmax, torch::nn::Softmax);
RIEMANN_FEATURE_ACTIVATION(RiemannLogSoftmax, torch::nn::LogSoftmax);
RIEMANN_FEATURE_ACTIVATION(RiemannDropout, torch::nn::Dropout);
RIEMANN_FEATURE_ACTIVATION(RiemannAlphaDropout, torch::nn::AlphaDropout);

#undef RIEMANN_FEATURE_ACTIVATION

class RiemannHardsigmoidImpl : public torch::nn::Module {
public:
  SparseTensor forward(const SparseTensor& input) {
    return SparseTensor(at::hardsigmoid(input.F()),
                        input.coordinate_map_key(),
                        input.coordinate_manager());
  }
};
TORCH_MODULE(RiemannHardsigmoid);

class RiemannHardswishImpl : public torch::nn::Module {
public:
  SparseTensor forward(const SparseTensor& input) {
    return SparseTensor(at::hardswish(input.F()), input.coordinate_map_key(),
                        input.coordinate_manager());
  }
};
TORCH_MODULE(RiemannHardswish);

class RiemannSinusoidalImpl : public torch::nn::Cloneable<RiemannSinusoidalImpl> {
public:
  RiemannSinusoidalImpl(int in_channels, int out_channels);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  int in_channels_;
  int out_channels_;
  torch::Tensor kernel_{nullptr};
  torch::Tensor bias_{nullptr};
  torch::Tensor coef_{nullptr};
};
TORCH_MODULE(RiemannSinusoidal);

} // namespace riemann

#endif
