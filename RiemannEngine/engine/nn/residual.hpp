#ifndef RIEMANN_ENGINE_NN_RESIDUAL_HPP
#define RIEMANN_ENGINE_NN_RESIDUAL_HPP

#include "engine/nn/activation.hpp"
#include "engine/nn/conv.hpp"
#include "engine/nn/normalization.hpp"

#include <torch/torch.h>

namespace riemann {

class BasicBlockImpl : public torch::nn::Cloneable<BasicBlockImpl> {
public:
  static constexpr int expansion = 1;
  BasicBlockImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                 torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                 double bn_momentum = 0.1, int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  RiemannConvolution conv1_{nullptr}, conv2_{nullptr};
  RiemannBatchNorm norm1_{nullptr}, norm2_{nullptr};
  RiemannReLU relu_{nullptr};
  torch::nn::AnyModule downsample_;
  int inplanes_, planes_, stride_, dilation_, dimension_;
  double bn_momentum_;
  bool has_downsample_ = false;
};
TORCH_MODULE(BasicBlock);

class BottleneckImpl : public torch::nn::Cloneable<BottleneckImpl> {
public:
  static constexpr int expansion = 4;
  BottleneckImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                 torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                 double bn_momentum = 0.1, int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  RiemannConvolution conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr};
  RiemannBatchNorm norm1_{nullptr}, norm2_{nullptr}, norm3_{nullptr};
  RiemannReLU relu_{nullptr};
  torch::nn::AnyModule downsample_;
  int inplanes_, planes_, stride_, dilation_, dimension_;
  double bn_momentum_;
  bool has_downsample_ = false;
};
TORCH_MODULE(Bottleneck);

} // namespace riemann

#endif
