#include "engine/nn/conv.hpp"

#include <cmath>
#include <stdexcept>

namespace riemann {
namespace {

at::Tensor evaluate_sparse_convolution(
    const SparseTensor& input, const at::Tensor& weights,
    const KernelGenerator::KernelParams& params,
    const CoordinateMapKey& output_key, bool transposed) {
  auto manager = input.coordinate_manager();
  auto features = input.F();
  if (features.is_cuda()) {
    const auto source_manager = manager;
    manager = std::make_shared<CoordinateManager>(
        input.dimension(), CoordinateMapBackend::CPU);
    const auto input_key = input.coordinate_map_key().get_key();
    const auto input_coordinates =
        source_manager->get_coordinates(input.coordinate_map_key())
            .to(torch::kCPU)
            .contiguous();
    const auto inserted_input = manager->insert_and_map(
        input_coordinates, input_key.first, input_key.second);
    features = features.index_select(
        0, inserted_input.second.first.to(features.device(), torch::kLong));

    if (!(input.coordinate_map_key() == output_key)) {
      const auto output_coordinates =
          source_manager->get_coordinates(output_key)
              .to(torch::kCPU)
              .contiguous();
      const auto output_key_value = output_key.get_key();
      manager->insert_and_map(output_coordinates, output_key_value.first,
                              output_key_value.second);
    }
  }
  auto* cpu_manager = manager->manager_ptr<cpu_manager_type<int32_t>>();
  const auto& map = cpu_manager->kernel_map(
      &input.coordinate_map_key(), &output_key, params.kernel_size,
      params.kernel_stride, params.kernel_dilation, params.region_type,
      params.region_offsets, transposed, false);

  const auto output_count = static_cast<int64_t>(manager->size(output_key));
  auto result = at::zeros({output_count, weights.size(2)}, input.F().options());
  for (size_t offset = 0; offset < map.first.size(); ++offset) {
    const auto& in_rows = map.first[offset];
    const auto& out_rows = map.second[offset];
    if (in_rows.empty())
      continue;
    auto in_index = torch::from_blob(
        const_cast<index_type*>(in_rows.data()),
        {static_cast<int64_t>(in_rows.size())},
        torch::TensorOptions().dtype(torch::kInt32))
                       .to(features.device(), torch::kLong);
    auto out_index = torch::from_blob(
        const_cast<index_type*>(out_rows.data()),
        {static_cast<int64_t>(out_rows.size())},
        torch::TensorOptions().dtype(torch::kInt32))
                        .to(features.device(), torch::kLong);
    auto values = features.index_select(0, in_index).matmul(weights[offset]);
    result.index_add_(0, out_index, values);
  }
  return result;
}

} // namespace

RiemannConvolutionBaseImpl::RiemannConvolutionBaseImpl(
    int in_channels, int out_channels, int kernel_size, int stride,
    int dilation, bool bias, bool is_transpose, bool expand_coordinates,
    ConvolutionMode::Type convolution_mode, int dimension,
    RegionType::Type region_type)
    : in_channels_(in_channels), out_channels_(out_channels),
      is_transpose_(is_transpose), use_mm_(false), dimension_(dimension),
      convolution_mode_(convolution_mode),
      kernel_generator_({kernel_size}, {stride}, {dilation}, is_transpose,
                        region_type, {}, expand_coordinates, {}, dimension) {
  TORCH_CHECK(in_channels > 0 && out_channels > 0,
              "Convolution channel counts must be positive");
  const auto volume = kernel_generator_.kernel_volume();
  kernel_ = register_parameter(
      "kernel", torch::empty({volume, in_channels_, out_channels_}));
  if (bias)
    bias_ = register_parameter("bias", torch::empty({1, out_channels_}));
  reset_parameters(is_transpose);
}

void RiemannConvolutionBaseImpl::reset() {
  reset_parameters(is_transpose_);
}

void RiemannConvolutionBaseImpl::reset_parameters(bool is_transpose) {
  torch::NoGradGuard guard;
  const auto fan = (is_transpose ? out_channels_ : in_channels_) *
                   kernel_generator_.kernel_volume();
  const auto bound = 1.0 / std::sqrt(static_cast<double>(fan));
  kernel_.uniform_(-bound, bound);
  if (bias_.defined())
    bias_.uniform_(-bound, bound);
}

SparseTensor RiemannConvolutionBaseImpl::forward(const SparseTensor& input) {
  TORCH_CHECK(input.dimension() == dimension_, "Input dimension mismatch");
  TORCH_CHECK(input.F().size(1) == in_channels_,
              "Input channel count does not match convolution");

  const auto params =
      kernel_generator_.get_kernel(input.tensor_stride(), is_transpose_);
  const auto input_key = input.coordinate_map_key().get_key();
  auto output_stride = input_key.first;
  for (size_t axis = 0; axis < output_stride.size(); ++axis) {
    const auto step = params.kernel_stride[axis];
    if (is_transpose_) {
      TORCH_CHECK(step != 0 && output_stride[axis] % step == 0,
                  "Transpose convolution stride must divide tensor stride");
      output_stride[axis] /= step;
    } else {
      output_stride[axis] *= step;
    }
  }
  CoordinateMapKey output_key(
      static_cast<size_type>(input.coordinate_map_key().get_coordinate_size()),
      {output_stride, input_key.second});
  auto manager = input.coordinate_manager();
  if (!is_transpose_ &&
      params.kernel_stride != stride_type(dimension_, 1)) {
    const auto strided = manager->stride(input.coordinate_map_key(),
                                         params.kernel_stride);
    output_key = std::move(strided);
  }
  auto result = evaluate_sparse_convolution(input, kernel_, params, output_key,
                                            is_transpose_);
  if (bias_.defined())
    result = result + bias_;
  return SparseTensor(std::move(result), output_key, std::move(manager));
}

RiemannConvolutionImpl::RiemannConvolutionImpl(
    int in_channels, int out_channels, int kernel_size, int stride,
    int dilation, bool bias, bool expand_coordinates,
    ConvolutionMode::Type convolution_mode, int dimension) {
  base_ = std::make_shared<RiemannConvolutionBaseImpl>(
      in_channels, out_channels, kernel_size, stride, dilation, bias, false,
      expand_coordinates, convolution_mode, dimension);
  register_module("base", base_);
}

void RiemannConvolutionImpl::reset() { base_->reset(); }

SparseTensor RiemannConvolutionImpl::forward(const SparseTensor& input) {
  return base_->forward(input);
}

RiemannConvolutionTransposeImpl::RiemannConvolutionTransposeImpl(
    int in_channels, int out_channels, int kernel_size, int stride,
    int dilation, bool bias, bool expand_coordinates,
    ConvolutionMode::Type convolution_mode, int dimension) {
  base_ = std::make_shared<RiemannConvolutionBaseImpl>(
      in_channels, out_channels, kernel_size, stride, dilation, bias, true,
      expand_coordinates, convolution_mode, dimension);
  register_module("base", base_);
}

void RiemannConvolutionTransposeImpl::reset() { base_->reset(); }

SparseTensor
RiemannConvolutionTransposeImpl::forward(const SparseTensor& input) {
  return base_->forward(input);
}

RiemannGenerativeConvolutionTransposeImpl::
    RiemannGenerativeConvolutionTransposeImpl(
        int in_channels, int out_channels, int kernel_size, int stride,
        int dilation, bool bias, ConvolutionMode::Type convolution_mode,
        int dimension) {
  base_ = std::make_shared<RiemannConvolutionBaseImpl>(
      in_channels, out_channels, kernel_size, stride, dilation, bias, true,
      true, convolution_mode, dimension);
  register_module("base", base_);
}

void RiemannGenerativeConvolutionTransposeImpl::reset() { base_->reset(); }

SparseTensor RiemannGenerativeConvolutionTransposeImpl::forward(
    const SparseTensor& input) {
  return base_->forward(input);
}

} // namespace riemann
