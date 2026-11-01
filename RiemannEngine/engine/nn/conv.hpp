#ifndef RIEMANN_ENGINE_NN_CONV_HPP
#define RIEMANN_ENGINE_NN_CONV_HPP

#include "engine/tensor/sparse_tensor.hpp"
#include "engine/grid/stencil.hpp"

#include <torch/torch.h>

#include <memory>

namespace riemann {

class RiemannConvolutionBaseImpl
    : public torch::nn::Cloneable<RiemannConvolutionBaseImpl> {
public:
  RiemannConvolutionBaseImpl(
      int in_channels, int out_channels, int kernel_size = 3,
      int stride = 1, int dilation = 1, bool bias = false,
      bool is_transpose = false, bool expand_coordinates = false,
      ConvolutionMode::Type convolution_mode = ConvolutionMode::DEFAULT,
      int dimension = 3,
      RegionType::Type region_type = RegionType::HYPER_CUBE);

  void reset() override;
  void reset_parameters(bool is_transpose = false);
  SparseTensor forward(const SparseTensor& input);

  int in_channels_;
  int out_channels_;
  bool is_transpose_;
  bool use_mm_;
  int dimension_;
  ConvolutionMode::Type convolution_mode_;
  KernelGenerator kernel_generator_;
  torch::Tensor kernel_{nullptr};
  torch::Tensor bias_{nullptr};
};

class RiemannConvolutionImpl
    : public torch::nn::Cloneable<RiemannConvolutionImpl> {
public:
  RiemannConvolutionImpl(int in_channels, int out_channels,
                         int kernel_size = 3, int stride = 1,
                         int dilation = 1, bool bias = false,
                         bool expand_coordinates = false,
                         ConvolutionMode::Type convolution_mode =
                             ConvolutionMode::DEFAULT,
                         int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);
  std::shared_ptr<RiemannConvolutionBaseImpl> base() { return base_; }

private:
  std::shared_ptr<RiemannConvolutionBaseImpl> base_{nullptr};
};
TORCH_MODULE(RiemannConvolution);

class RiemannConvolutionTransposeImpl
    : public torch::nn::Cloneable<RiemannConvolutionTransposeImpl> {
public:
  RiemannConvolutionTransposeImpl(
      int in_channels, int out_channels, int kernel_size = 3, int stride = 1,
      int dilation = 1, bool bias = false, bool expand_coordinates = false,
      ConvolutionMode::Type convolution_mode = ConvolutionMode::DEFAULT,
      int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);
  std::shared_ptr<RiemannConvolutionBaseImpl> base() { return base_; }

private:
  std::shared_ptr<RiemannConvolutionBaseImpl> base_{nullptr};
};
TORCH_MODULE(RiemannConvolutionTranspose);

class RiemannGenerativeConvolutionTransposeImpl
    : public torch::nn::Cloneable<RiemannGenerativeConvolutionTransposeImpl> {
public:
  RiemannGenerativeConvolutionTransposeImpl(
      int in_channels, int out_channels, int kernel_size = 3, int stride = 1,
      int dilation = 1, bool bias = false,
      ConvolutionMode::Type convolution_mode = ConvolutionMode::DEFAULT,
      int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor& input);
  std::shared_ptr<RiemannConvolutionBaseImpl> base() { return base_; }

private:
  std::shared_ptr<RiemannConvolutionBaseImpl> base_{nullptr};
};
TORCH_MODULE(RiemannGenerativeConvolutionTranspose);

} // namespace riemann

#endif
