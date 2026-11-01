#ifndef RIEMANN_ENGINE_AUTODIFF_COMPAT_HPP
#define RIEMANN_ENGINE_AUTODIFF_COMPAT_HPP

#include "engine/nn/conv.hpp"
#include "engine/nn/normalization.hpp"
#include "engine/nn/operations.hpp"
#include "engine/nn/pooling.hpp"
#include "engine/nn/resample.hpp"

#include <torch/torch.h>

#include <memory>
#include <utility>
#include <vector>

namespace riemann {
namespace autodiff_compat_detail {

inline SparseTensor make_sparse(const at::Tensor& features,
                                const std::shared_ptr<CoordinateMapKey>& key,
                                const std::shared_ptr<CoordinateManager>& manager) {
  TORCH_CHECK(key != nullptr, "Coordinate map key must not be null");
  TORCH_CHECK(manager != nullptr, "Coordinate manager must not be null");
  TORCH_CHECK(key->is_key_set(), "Coordinate map key must be set");
  return SparseTensor(features, *key, manager);
}

inline void update_key(const std::shared_ptr<CoordinateMapKey>& key,
                       const SparseTensor& output) {
  TORCH_CHECK(key != nullptr, "Output coordinate map key must not be null");
  key->set_key(output.coordinate_map_key().get_key());
}

inline std::vector<int64_t> to_int64(const stride_type& values) {
  return std::vector<int64_t>(values.begin(), values.end());
}

inline int dimension_of(const std::shared_ptr<CoordinateMapKey>& key) {
  TORCH_CHECK(key != nullptr, "Coordinate map key must not be null");
  return static_cast<int>(key->get_coordinate_size()) - 1;
}

inline std::pair<at::Tensor, at::Tensor>
max_pool_forward(const at::Tensor& input_map, const at::Tensor& output_map,
                 const at::Tensor& input_features, int64_t output_rows) {
  TORCH_CHECK(input_map.numel() == output_map.numel(),
              "Pooling map arrays must have equal lengths");
  TORCH_CHECK(input_features.dim() == 2 && output_rows >= 0,
              "Invalid max-pooling input shape");

  auto input_rows = input_map.reshape({-1}).to(input_features.device(), at::kLong);
  auto output_indices =
      output_map.reshape({-1}).to(input_features.device(), at::kLong);
  std::vector<at::Tensor> values;
  std::vector<at::Tensor> masks;
  values.reserve(static_cast<size_t>(output_rows));
  masks.reserve(static_cast<size_t>(output_rows));
  for (int64_t output_row = 0; output_row < output_rows; ++output_row) {
    auto matches = at::nonzero(output_indices == output_row).reshape({-1});
    if (matches.numel() == 0) {
      values.push_back(torch::zeros({input_features.size(1)},
                                    input_features.options()));
      masks.push_back(torch::full({input_features.size(1)}, -1,
                                  input_map.options()));
      continue;
    }
    auto source_rows = input_rows.index_select(0, matches);
    auto candidates = input_features.index_select(0, source_rows);
    auto maximum = candidates.max(0);
    values.push_back(std::get<0>(maximum));
    masks.push_back(source_rows.index_select(0, std::get<1>(maximum))
                        .to(input_map.options()));
  }
  auto output = output_rows == 0
                    ? torch::empty({0, input_features.size(1)},
                                   input_features.options())
                    : torch::stack(values, 0);
  auto mask = output_rows == 0
                  ? torch::empty({0, input_features.size(1)}, input_map.options())
                  : torch::stack(masks, 0);
  return {std::move(output), std::move(mask)};
}

} // namespace autodiff_compat_detail

class ConvolutionFunction
    : public torch::autograd::Function<ConvolutionFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, at::Tensor kernel, stride_type kernel_size,
      stride_type kernel_stride, stride_type kernel_dilation,
      RegionType::Type region_type, at::Tensor region_offsets,
      bool expand_coordinates, ConvolutionMode::Type conv_mode,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager) {
    TORCH_CHECK(kernel.dim() == 3,
                "Convolution kernel must have shape [kernel, in, out]");
    TORCH_CHECK(kernel.size(1) == input_features.size(1),
                "Convolution kernel input channels do not match features");
    const auto dimension = autodiff_compat_detail::dimension_of(in_key);
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    RiemannConvolutionBaseImpl layer(
        static_cast<int>(input_features.size(1)),
        static_cast<int>(kernel.size(2)), 1, 1, 1, false, false,
        expand_coordinates, conv_mode, dimension, region_type);
    layer.kernel_generator_ = KernelGenerator(
        autodiff_compat_detail::to_int64(kernel_size),
        autodiff_compat_detail::to_int64(kernel_stride),
        autodiff_compat_detail::to_int64(kernel_dilation), false, region_type,
        std::move(region_offsets), expand_coordinates, {}, dimension);
    TORCH_CHECK(kernel.size(0) == layer.kernel_generator_.kernel_volume(),
                "Convolution kernel volume does not match its geometry");
    layer.kernel_ = std::move(kernel);
    auto output = layer.forward(input);
    autodiff_compat_detail::update_key(out_key, output);
    return output.F();
  }
};

class ConvolutionTransposeFunction
    : public torch::autograd::Function<ConvolutionTransposeFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, at::Tensor kernel, stride_type kernel_size,
      stride_type kernel_stride, stride_type kernel_dilation,
      RegionType::Type region_type, at::Tensor region_offsets,
      bool expand_coordinates, ConvolutionMode::Type conv_mode,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager) {
    TORCH_CHECK(kernel.dim() == 3,
                "Convolution kernel must have shape [kernel, in, out]");
    TORCH_CHECK(kernel.size(1) == input_features.size(1),
                "Convolution kernel input channels do not match features");
    const auto dimension = autodiff_compat_detail::dimension_of(in_key);
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    RiemannConvolutionBaseImpl layer(
        static_cast<int>(input_features.size(1)),
        static_cast<int>(kernel.size(2)), 1, 1, 1, false, true,
        expand_coordinates, conv_mode, dimension, region_type);
    layer.kernel_generator_ = KernelGenerator(
        autodiff_compat_detail::to_int64(kernel_size),
        autodiff_compat_detail::to_int64(kernel_stride),
        autodiff_compat_detail::to_int64(kernel_dilation), true, region_type,
        std::move(region_offsets), expand_coordinates, {}, dimension);
    TORCH_CHECK(kernel.size(0) == layer.kernel_generator_.kernel_volume(),
                "Convolution kernel volume does not match its geometry");
    layer.kernel_ = std::move(kernel);
    auto output = layer.forward(input);
    autodiff_compat_detail::update_key(out_key, output);
    return output.F();
  }
};

class LocalPoolingFunction
    : public torch::autograd::Function<LocalPoolingFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, PoolingMode::Type pooling_mode,
      stride_type kernel_size, stride_type kernel_stride,
      stride_type kernel_dilation, RegionType::Type region_type,
      at::Tensor region_offsets, std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager) {
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    KernelGenerator generator(
        autodiff_compat_detail::to_int64(kernel_size),
        autodiff_compat_detail::to_int64(kernel_stride),
        autodiff_compat_detail::to_int64(kernel_dilation), false, region_type,
        std::move(region_offsets), false, {}, input.dimension());
    RiemannPoolingBaseImpl layer(std::move(generator), pooling_mode,
                                 input.dimension());
    auto output = layer.forward(input);
    autodiff_compat_detail::update_key(out_key, output);
    return output.F();
  }
};

class LocalPoolingTransposeFunction
    : public torch::autograd::Function<LocalPoolingTransposeFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, PoolingMode::Type pooling_mode,
      stride_type kernel_size, stride_type kernel_stride,
      stride_type kernel_dilation, RegionType::Type region_type,
      at::Tensor region_offsets, bool expand_coordinates,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager) {
    TORCH_CHECK(pooling_mode == PoolingMode::LOCAL_SUM_POOLING ||
                    pooling_mode == PoolingMode::LOCAL_AVG_POOLING ||
                    pooling_mode == PoolingMode::LOCAL_MAX_POOLING,
                "Transpose pooling requires a local pooling mode");
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    KernelGenerator generator(
        autodiff_compat_detail::to_int64(kernel_size),
        autodiff_compat_detail::to_int64(kernel_stride),
        autodiff_compat_detail::to_int64(kernel_dilation), true, region_type,
        std::move(region_offsets), expand_coordinates, {}, input.dimension());
    RiemannPoolingTransposeImpl layer(std::move(generator), input.dimension(),
                                      pooling_mode);
    auto output = layer.forward(input);
    autodiff_compat_detail::update_key(out_key, output);
    return output.F();
  }
};

class GlobalPoolingFunction
    : public torch::autograd::Function<GlobalPoolingFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, PoolingMode::Type pooling_mode,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager) {
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    auto output = RiemannGlobalPooling(pooling_mode)->forward(input);
    autodiff_compat_detail::update_key(out_key, output);
    return output.F();
  }
};

class DirectMaxPoolingFunction
    : public torch::autograd::Function<DirectMaxPoolingFunction> {
public:
  static at::Tensor apply(at::Tensor input_map, at::Tensor output_map,
                          at::Tensor input_features, int64_t output_rows,
                          bool /*is_sorted*/) {
    return autodiff_compat_detail::max_pool_forward(
               input_map, output_map, input_features, output_rows)
        .first;
  }
};

inline std::pair<at::Tensor, at::Tensor>
max_pool_fw(const at::Tensor& input_map, const at::Tensor& output_map,
            const at::Tensor& input_features, int output_rows,
            bool /*is_sorted*/) {
  return autodiff_compat_detail::max_pool_forward(
      input_map, output_map, input_features, output_rows);
}

inline at::Tensor max_pool_bw(const at::Tensor& grad_output,
                              const at::Tensor& mask_index, int input_rows) {
  TORCH_CHECK(grad_output.dim() == 2 && mask_index.sizes() == grad_output.sizes(),
              "Max-pooling gradient and mask shapes must match");
  TORCH_CHECK(input_rows >= 0, "Input row count must be non-negative");
  const auto channels = grad_output.size(1);
  auto source_indices = mask_index.reshape({-1}).to(grad_output.device(), at::kLong);
  auto feature_indices = torch::arange(channels, source_indices.options())
                             .repeat(grad_output.size(0));
  auto valid = source_indices >= 0;
  TORCH_CHECK((source_indices.masked_select(valid) < input_rows)
                  .all().item<bool>(),
              "Max-pooling mask index is outside the input rows");
  source_indices = source_indices.masked_select(valid);
  feature_indices = feature_indices.masked_select(valid);
  auto gradients = grad_output.reshape({-1}).masked_select(valid);
  auto flattened_indices = source_indices * channels + feature_indices;
  auto output = torch::zeros({input_rows * channels}, grad_output.options());
  output = output.index_add(0, flattened_indices, gradients);
  return output.reshape({input_rows, channels});
}

class BroadcastFunction
    : public torch::autograd::Function<BroadcastFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, at::Tensor input_global_features,
      BroadcastMode::Type operation_type,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> global_key,
      std::shared_ptr<CoordinateManager> manager) {
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    auto global = autodiff_compat_detail::make_sparse(
        input_global_features, global_key, manager);
    return RiemannBroadcastBaseImpl(operation_type)
        .forward(input, global)
        .F();
  }
};

class PruningFunction : public torch::autograd::Function<PruningFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, at::Tensor mask,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager) {
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    auto output = RiemannPruning()->forward(input, mask);
    autodiff_compat_detail::update_key(out_key, output);
    return output.F();
  }
};

class InterpolationFunction
    : public torch::autograd::Function<InterpolationFunction> {
public:
  static torch::autograd::variable_list apply(
      at::Tensor input_features, at::Tensor field,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateManager> manager) {
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    const auto result =
        RiemannInterpolation()->forward_full(input, field);
    return {result.output, result.in_map, result.out_map, result.weights};
  }
};

class InstanceNormFunction
    : public torch::autograd::Function<InstanceNormFunction> {
public:
  static at::Tensor apply(
      at::Tensor input_features, std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> global_key,
      std::shared_ptr<CoordinateManager> manager,
      PoolingMode::Type global_pooling_mode =
          PoolingMode::GLOBAL_AVG_POOLING_KERNEL) {
    auto input =
        autodiff_compat_detail::make_sparse(input_features, in_key, manager);
    auto pooled =
        RiemannGlobalPooling(global_pooling_mode)->forward(input);
    autodiff_compat_detail::update_key(global_key, pooled);
    auto centered_global =
        SparseTensor(-pooled.F(), pooled.coordinate_map_key(), manager);
    auto centered = RiemannBroadcastBaseImpl(
                        BroadcastMode::ELEMENTWISE_ADDITON)
                        .forward(input, centered_global);
    auto variance = RiemannGlobalPooling(global_pooling_mode)
                        ->forward(SparseTensor(centered.F().square(),
                                               centered.coordinate_map_key(),
                                               manager));
    auto inverse_std =
        SparseTensor(at::rsqrt(variance.F() + 1e-8),
                     variance.coordinate_map_key(), manager);
    return RiemannBroadcastBaseImpl(
               BroadcastMode::ELEMENTWISE_MULTIPLICATION)
        .forward(centered, inverse_std)
        .F();
  }
};

} // namespace riemann

#endif
