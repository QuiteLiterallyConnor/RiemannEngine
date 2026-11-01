#ifndef RIEMANN_ENGINE_TENSOR_SPARSE_TENSOR_HPP
#define RIEMANN_ENGINE_TENSOR_SPARSE_TENSOR_HPP

#include "engine/foundation/types.hpp"
#include "engine/grid/key.hpp"
#include "engine/grid/space.hpp"

#include <torch/torch.h>

#include <memory>
#include <utility>
#include <vector>

namespace riemann {

class SparseTensor {
public:
  SparseTensor(
      at::Tensor features, at::Tensor coordinates,
      std::shared_ptr<CoordinateManager> manager = nullptr,
      SparseTensorQuantizationMode::Type quantization_mode =
          SparseTensorQuantizationMode::RANDOM_SUBSAMPLE,
      GPUMemoryAllocatorBackend::Type alloc_type =
          GPUMemoryAllocatorBackend::PYTORCH,
      RiemannAlgorithm::Mode algo = RiemannAlgorithm::DEFAULT);

  SparseTensor(at::Tensor features, CoordinateMapKey coordinate_map_key,
               std::shared_ptr<CoordinateManager> manager);

  const at::Tensor& F() const noexcept { return features_; }
  at::Tensor C() const;
  const CoordinateMapKey& coordinate_map_key() const noexcept { return key_; }
  std::shared_ptr<CoordinateManager> coordinate_manager() const noexcept {
    return manager_;
  }
  stride_type tensor_stride() const;
  int dimension() const noexcept { return dimension_; }
  torch::Device device() const { return features_.device(); }
  c10::ScalarType dtype() const { return features_.scalar_type(); }
  bool requires_grad() const { return features_.requires_grad(); }

  std::vector<at::Tensor> decomposed_coordinates() const;
  std::vector<at::Tensor> decomposed_features() const;
  std::vector<std::pair<at::Tensor, at::Tensor>>
  decomposed_coordinates_and_features() const;

  at::Tensor dense(
      c10::optional<std::vector<int64_t>> shape = c10::nullopt,
      c10::optional<at::Tensor> min_coordinate = c10::nullopt) const;
  at::Tensor sparse() const;

  SparseTensor operator+(const SparseTensor& other) const;
  SparseTensor operator-(const SparseTensor& other) const;
  SparseTensor operator*(const SparseTensor& other) const;
  SparseTensor operator/(const SparseTensor& other) const;
  SparseTensor& operator+=(const SparseTensor& other);
  SparseTensor operator+(at::Scalar scalar) const;
  SparseTensor operator*(at::Scalar scalar) const;

  SparseTensor to(torch::Device device) const;
  SparseTensor to(c10::ScalarType dtype) const;
  SparseTensor float_() const;
  SparseTensor double_() const;
  SparseTensor detach() const;

private:
  at::Tensor quantize_features(const at::Tensor& features,
                               const at::Tensor& first_rows,
                               const at::Tensor& row_to_unique,
                               int64_t unique_count,
                               SparseTensorQuantizationMode::Type mode) const;
  void require_compatible(const SparseTensor& other,
                          const char* operation) const;

  at::Tensor features_;
  mutable c10::optional<at::Tensor> coordinates_;
  CoordinateMapKey key_;
  std::shared_ptr<CoordinateManager> manager_;
  int dimension_;
};

} // namespace riemann

#endif
