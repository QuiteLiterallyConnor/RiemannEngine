#ifndef RIEMANN_ENGINE_NN_SQUEEZE_EXCITE_HPP
#define RIEMANN_ENGINE_NN_SQUEEZE_EXCITE_HPP

#include "engine/nn/linear.hpp"
#include "engine/nn/pooling.hpp"
#include "engine/nn/residual.hpp"
#include "engine/nn/operations.hpp"

namespace riemann {

class SELayerImpl : public torch::nn::Cloneable<SELayerImpl> {
public:
  SELayerImpl(int channel, int reduction = 16, int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  RiemannLinear fc1_{nullptr}, fc2_{nullptr};
  RiemannReLU fc_relu_{nullptr};
  RiemannSigmoid fc_sigmoid_{nullptr};
  RiemannGlobalPooling pooling_{nullptr};
  RiemannBroadcastMultiplication broadcast_mul_{nullptr};

private:
  int channel_, reduction_, dimension_;
};
TORCH_MODULE(SELayer);

class SEBasicBlockImpl : public torch::nn::Cloneable<SEBasicBlockImpl> {
public:
  static constexpr int expansion = 1;
  SEBasicBlockImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                   torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                   int reduction = 16, double bn_momentum = 0.1,
                   int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  RiemannConvolution conv1_{nullptr}, conv2_{nullptr};
  RiemannBatchNorm norm1_{nullptr}, norm2_{nullptr};
  RiemannReLU relu_{nullptr};
  SELayer se_{nullptr};
  int inplanes_, planes_, stride_, dilation_, reduction_, dimension_;
  double bn_momentum_;
  torch::nn::AnyModule downsample_;
  bool has_downsample_ = false;
};
TORCH_MODULE(SEBasicBlock);

class SEBottleneckImpl : public torch::nn::Cloneable<SEBottleneckImpl> {
public:
  static constexpr int expansion = 4;
  SEBottleneckImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                   torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                   int reduction = 16, double bn_momentum = 0.1,
                   int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);

  RiemannConvolution conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr};
  RiemannBatchNorm norm1_{nullptr}, norm2_{nullptr}, norm3_{nullptr};
  RiemannReLU relu_{nullptr};
  SELayer se_{nullptr};
  int inplanes_, planes_, stride_, dilation_, reduction_, dimension_;
  double bn_momentum_;
  torch::nn::AnyModule downsample_;
  bool has_downsample_ = false;
};
TORCH_MODULE(SEBottleneck);

} // namespace riemann

#endif
