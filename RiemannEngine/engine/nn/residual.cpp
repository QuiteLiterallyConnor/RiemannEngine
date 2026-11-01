#include "engine/nn/residual.hpp"

namespace riemann {
namespace {

RiemannConvolution make_convolution(int in_channels, int out_channels,
                                    int kernel_size, int stride, int dilation,
                                    int dimension) {
  return RiemannConvolution(in_channels, out_channels, kernel_size, stride,
                            dilation, false, false, ConvolutionMode::DEFAULT,
                            dimension);
}

} // namespace

BasicBlockImpl::BasicBlockImpl(int inplanes, int planes, int stride,
                               int dilation, torch::nn::AnyModule downsample,
                               double bn_momentum, int dimension)
    : downsample_(std::move(downsample)), inplanes_(inplanes), planes_(planes),
      stride_(stride), dilation_(dilation), dimension_(dimension),
      bn_momentum_(bn_momentum), has_downsample_(!downsample_.is_empty()) {
  TORCH_CHECK(inplanes > 0 && planes > 0, "Block channels must be positive");
  TORCH_CHECK(stride > 0 && dilation > 0,
              "Block stride and dilation must be positive");
  TORCH_CHECK(dimension > 0, "Dimension must be positive");
  reset();
  if (has_downsample_)
    register_module("downsample", downsample_.ptr());
}

void BasicBlockImpl::reset() {
  conv1_ = register_module(
      "conv1", make_convolution(inplanes_, planes_, 3, stride_, dilation_,
                                dimension_));
  norm1_ = register_module(
      "norm1", RiemannBatchNorm(planes_, 1e-5, bn_momentum_));
  conv2_ = register_module(
      "conv2", make_convolution(planes_, planes_, 3, 1, dilation_, dimension_));
  norm2_ = register_module(
      "norm2", RiemannBatchNorm(planes_, 1e-5, bn_momentum_));
  relu_ = register_module("relu", RiemannReLU(true));
}

SparseTensor BasicBlockImpl::forward(const SparseTensor& input) {
  auto residual = input;
  auto output = relu_->forward(norm1_->forward(conv1_->forward(input)));
  output = norm2_->forward(conv2_->forward(output));
  if (has_downsample_)
    residual = downsample_.forward<SparseTensor>(input);
  output += residual;
  return relu_->forward(output);
}

BottleneckImpl::BottleneckImpl(int inplanes, int planes, int stride,
                               int dilation, torch::nn::AnyModule downsample,
                               double bn_momentum, int dimension)
    : downsample_(std::move(downsample)), inplanes_(inplanes), planes_(planes),
      stride_(stride), dilation_(dilation), dimension_(dimension),
      bn_momentum_(bn_momentum), has_downsample_(!downsample_.is_empty()) {
  TORCH_CHECK(inplanes > 0 && planes > 0, "Block channels must be positive");
  TORCH_CHECK(stride > 0 && dilation > 0,
              "Block stride and dilation must be positive");
  TORCH_CHECK(dimension > 0, "Dimension must be positive");
  reset();
  if (has_downsample_)
    register_module("downsample", downsample_.ptr());
}

void BottleneckImpl::reset() {
  conv1_ = register_module(
      "conv1", make_convolution(inplanes_, planes_, 1, 1, 1, dimension_));
  norm1_ = register_module(
      "norm1", RiemannBatchNorm(planes_, 1e-5, bn_momentum_));
  conv2_ = register_module(
      "conv2", make_convolution(planes_, planes_, 3, stride_, dilation_,
                                dimension_));
  norm2_ = register_module(
      "norm2", RiemannBatchNorm(planes_, 1e-5, bn_momentum_));
  conv3_ = register_module(
      "conv3", make_convolution(planes_, planes_ * expansion, 1, 1, 1,
                                dimension_));
  norm3_ = register_module(
      "norm3", RiemannBatchNorm(planes_ * expansion, 1e-5, bn_momentum_));
  relu_ = register_module("relu", RiemannReLU(true));
}

SparseTensor BottleneckImpl::forward(const SparseTensor& input) {
  auto residual = input;
  auto output = relu_->forward(norm1_->forward(conv1_->forward(input)));
  output = relu_->forward(norm2_->forward(conv2_->forward(output)));
  output = norm3_->forward(conv3_->forward(output));
  if (has_downsample_)
    residual = downsample_.forward<SparseTensor>(input);
  output += residual;
  return relu_->forward(output);
}

} // namespace riemann
