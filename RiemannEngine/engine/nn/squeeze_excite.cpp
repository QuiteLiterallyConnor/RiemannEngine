#include "engine/nn/squeeze_excite.hpp"

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

SELayerImpl::SELayerImpl(int channel, int reduction, int dimension)
    : channel_(channel), reduction_(reduction), dimension_(dimension) {
  TORCH_CHECK(channel > 0 && reduction > 0,
              "SE channels and reduction must be positive");
  TORCH_CHECK(dimension > 0, "Dimension must be positive");
  TORCH_CHECK(channel / reduction > 0,
              "SE channel / reduction must be greater than zero");
  reset();
}

void SELayerImpl::reset() {
  const auto reduced = channel_ / reduction_;
  fc1_ = register_module("fc1", RiemannLinear(channel_, reduced, true));
  fc_relu_ = register_module("fc_relu", RiemannReLU(true));
  fc2_ = register_module("fc2", RiemannLinear(reduced, channel_, true));
  fc_sigmoid_ = register_module("fc_sigmoid", RiemannSigmoid());
  pooling_ = register_module("pooling", RiemannGlobalPooling());
  broadcast_mul_ =
      register_module("broadcast_mul", RiemannBroadcastMultiplication());
}

SparseTensor SELayerImpl::forward(const SparseTensor& input) {
  auto global = pooling_->forward(input);
  global = fc1_->forward(global);
  global = fc_relu_->forward(global);
  global = fc2_->forward(global);
  global = fc_sigmoid_->forward(global);
  return broadcast_mul_->forward(input, global);
}

SEBasicBlockImpl::SEBasicBlockImpl(
    int inplanes, int planes, int stride, int dilation,
    torch::nn::AnyModule downsample, int reduction, double bn_momentum,
    int dimension)
    : inplanes_(inplanes), planes_(planes), stride_(stride),
      dilation_(dilation), reduction_(reduction), dimension_(dimension),
      bn_momentum_(bn_momentum), downsample_(std::move(downsample)),
      has_downsample_(!downsample_.is_empty()) {
  TORCH_CHECK(inplanes > 0 && planes > 0 && reduction > 0,
              "Block channels and reduction must be positive");
  TORCH_CHECK(stride > 0 && dilation > 0 && dimension > 0,
              "Block stride, dilation, and dimension must be positive");
  reset();
  if (has_downsample_)
    register_module("downsample", downsample_.ptr());
}

void SEBasicBlockImpl::reset() {
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
  se_ = register_module("se", SELayer(planes_ * BasicBlockImpl::expansion,
                                       reduction_, dimension_));
}

SparseTensor SEBasicBlockImpl::forward(const SparseTensor& input) {
  auto residual = input;
  auto output = relu_->forward(norm1_->forward(conv1_->forward(input)));
  output = norm2_->forward(conv2_->forward(output));
  output = se_->forward(output);
  if (has_downsample_)
    residual = downsample_.forward<SparseTensor>(input);
  output += residual;
  return relu_->forward(output);
}

SEBottleneckImpl::SEBottleneckImpl(
    int inplanes, int planes, int stride, int dilation,
    torch::nn::AnyModule downsample, int reduction, double bn_momentum,
    int dimension)
    : inplanes_(inplanes), planes_(planes), stride_(stride),
      dilation_(dilation), reduction_(reduction), dimension_(dimension),
      bn_momentum_(bn_momentum), downsample_(std::move(downsample)),
      has_downsample_(!downsample_.is_empty()) {
  TORCH_CHECK(inplanes > 0 && planes > 0 && reduction > 0,
              "Block channels and reduction must be positive");
  TORCH_CHECK(stride > 0 && dilation > 0 && dimension > 0,
              "Block stride, dilation, and dimension must be positive");
  reset();
  if (has_downsample_)
    register_module("downsample", downsample_.ptr());
}

void SEBottleneckImpl::reset() {
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
  se_ = register_module(
      "se", SELayer(planes_ * expansion, reduction_,
                    dimension_));
}

SparseTensor SEBottleneckImpl::forward(const SparseTensor& input) {
  auto residual = input;
  auto output = relu_->forward(norm1_->forward(conv1_->forward(input)));
  output = relu_->forward(norm2_->forward(conv2_->forward(output)));
  output = norm3_->forward(conv3_->forward(output));
  output = se_->forward(output);
  if (has_downsample_)
    residual = downsample_.forward<SparseTensor>(input);
  output += residual;
  return relu_->forward(output);
}

} // namespace riemann
