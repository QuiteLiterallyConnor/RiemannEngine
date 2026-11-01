#ifndef RIEMANN_ENGINE_NN_NORMALIZATION_HPP
#define RIEMANN_ENGINE_NN_NORMALIZATION_HPP

#include "engine/nn/activation.hpp"
#include "engine/tensor/sparse_tensor.hpp"

#include <torch/torch.h>

namespace riemann {

class RiemannBatchNormImpl : public torch::nn::Module {
public:
  RiemannBatchNormImpl(
      int64_t num_features, double eps = 1e-5, double momentum = 0.1,
      bool affine = true, bool track_running_stats = true)
      : normalization_(register_module(
            "normalization",
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features)
                                       .eps(eps)
                                       .momentum(momentum)
                                       .affine(affine)
                                       .track_running_stats(
                                           track_running_stats)))) {}

  SparseTensor forward(const SparseTensor& input) {
    auto values = normalization_->forward(input.F());
    return SparseTensor(std::move(values), input.coordinate_map_key(),
                        input.coordinate_manager());
  }

private:
  torch::nn::BatchNorm1d normalization_{nullptr};
};
TORCH_MODULE(RiemannBatchNorm);

using RiemannSyncBatchNormImpl = RiemannBatchNormImpl;
using RiemannSyncBatchNorm = RiemannBatchNorm;

using RiemannLayerNormImpl =
    RiemannFeatureModuleImpl<torch::nn::LayerNorm>;
TORCH_MODULE(RiemannLayerNorm);

using RiemannGroupNormImpl =
    RiemannFeatureModuleImpl<torch::nn::GroupNorm>;
TORCH_MODULE(RiemannGroupNorm);

class RiemannInstanceNormImpl
    : public torch::nn::Cloneable<RiemannInstanceNormImpl> {
public:
  explicit RiemannInstanceNormImpl(int64_t num_features);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  int64_t num_features_;
  torch::Tensor weight_{nullptr};
  torch::Tensor bias_{nullptr};
  double eps_ = 1e-8;
};
TORCH_MODULE(RiemannInstanceNorm);

class RiemannStableInstanceNormImpl
    : public torch::nn::Cloneable<RiemannStableInstanceNormImpl> {
public:
  explicit RiemannStableInstanceNormImpl(int64_t num_features);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  int64_t num_features_;
  torch::Tensor weight_{nullptr};
  torch::Tensor bias_{nullptr};
  double eps_ = 1e-6;
};
TORCH_MODULE(RiemannStableInstanceNorm);

} // namespace riemann

#endif
