#ifndef RIEMANN_ENGINE_TENSOR_FIELD_HPP
#define RIEMANN_ENGINE_TENSOR_FIELD_HPP

#include "engine/tensor/sparse_tensor.hpp"

#include <unordered_map>

namespace riemann {

class TensorField {
public:
  TensorField(
      at::Tensor features, at::Tensor coordinates,
      std::shared_ptr<CoordinateManager> manager = nullptr,
      SparseTensorQuantizationMode::Type quantization_mode =
          SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE);

  SparseTensor sparse(
      stride_type tensor_stride = {},
      c10::optional<CoordinateMapKey> coordinate_map_key = c10::nullopt,
      SparseTensorQuantizationMode::Type quantization_mode =
          SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE) const;
  SparseTensor splat() const;
  at::Tensor inverse_mapping(const CoordinateMapKey& sparse_key) const;

  const at::Tensor& F() const noexcept { return features_; }
  at::Tensor C() const;
  const CoordinateMapKey& coordinate_field_map_key() const noexcept {
    return field_key_;
  }
  std::shared_ptr<CoordinateManager> coordinate_manager() const noexcept {
    return manager_;
  }
  int dimension() const noexcept { return dimension_; }
  torch::Device device() const { return features_.device(); }
  c10::ScalarType dtype() const { return features_.scalar_type(); }

private:
  at::Tensor quantize_features(const at::Tensor& first_rows,
                               const at::Tensor& row_to_unique,
                               int64_t unique_count,
                               SparseTensorQuantizationMode::Type mode) const;

  at::Tensor features_;
  mutable c10::optional<at::Tensor> coordinates_;
  CoordinateMapKey field_key_;
  std::shared_ptr<CoordinateManager> manager_;
  int dimension_;
  mutable std::unordered_map<coordinate_map_key_type, at::Tensor,
                             coordinate_map_key_hasher>
      inverse_mappings_;
};

} // namespace riemann

#endif
