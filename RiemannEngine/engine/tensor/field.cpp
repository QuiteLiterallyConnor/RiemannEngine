#include "engine/tensor/field.hpp"

#include <stdexcept>

namespace riemann {

TensorField::TensorField(
    at::Tensor features, at::Tensor coordinates,
    std::shared_ptr<CoordinateManager> manager,
    SparseTensorQuantizationMode::Type quantization_mode)
    : features_(std::move(features)),
      field_key_(static_cast<size_type>(coordinates.size(1))),
      dimension_(0) {
  (void)quantization_mode;
  TORCH_CHECK(features_.dim() == 2, "Features must be a 2D tensor");
  TORCH_CHECK(coordinates.dim() == 2, "Coordinates must be a 2D tensor");
  TORCH_CHECK(features_.size(0) == coordinates.size(0),
              "Features and coordinates must have same number of rows");
  TORCH_CHECK(coordinates.scalar_type() == at::kFloat ||
                  coordinates.scalar_type() == at::kDouble,
              "TensorField coordinates must be float or double");
  TORCH_CHECK(coordinates.size(1) >= 2,
              "Coordinates must contain a batch column and spatial columns");

  dimension_ = static_cast<int>(coordinates.size(1) - 1);
  if (!manager) {
    const auto backend = coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                                : CoordinateMapBackend::CPU;
    manager = std::make_shared<CoordinateManager>(dimension_, backend);
  }
  TORCH_CHECK(manager->D() == dimension_,
              "Coordinate manager dimension does not match coordinates");
  manager_ = std::move(manager);
  stride_type unit_stride(static_cast<size_t>(dimension_), 1);
  // Keep the original continuous coordinates: the manager may also hold a
  // quantized sparse map under the same key once sparse() has been called.
  coordinates_ = coordinates.contiguous();
  field_key_ = manager_->insert_field(coordinates_.value(), unit_stride);
}

SparseTensor TensorField::sparse(
    stride_type tensor_stride,
    c10::optional<CoordinateMapKey> coordinate_map_key,
    SparseTensorQuantizationMode::Type quantization_mode) const {
  (void)coordinate_map_key;
  if (tensor_stride.empty())
    tensor_stride.assign(static_cast<size_t>(dimension_), 1);

  auto inserted =
      manager_->field_to_sparse_insert_and_map(field_key_, tensor_stride);
  const auto& sparse_key = inserted.first;
  const auto& first_rows = inserted.second.first;
  const auto& row_to_unique = inserted.second.second;
  const auto count = static_cast<int64_t>(manager_->size(sparse_key));
  auto quantized =
      quantize_features(first_rows, row_to_unique, count, quantization_mode);
  inverse_mappings_[sparse_key.get_key()] = row_to_unique;
  return SparseTensor(std::move(quantized), sparse_key, manager_);
}

SparseTensor TensorField::splat() const {
  const auto coordinates = C();
  std::vector<at::Tensor> lattice_points;
  std::vector<at::Tensor> weight_parts;
  const auto corners = int64_t{1} << dimension_;
  const auto floor_coords = at::floor(coordinates);
  const auto fraction = coordinates - floor_coords;
  const auto integer_base = floor_coords.to(at::kInt);
  lattice_points.reserve(static_cast<size_t>(corners));
  weight_parts.reserve(static_cast<size_t>(corners));

  for (int64_t corner = 0; corner < corners; ++corner) {
    std::vector<int64_t> offset(static_cast<size_t>(dimension_ + 1), 0);
    auto weight = at::ones({coordinates.size(0)}, features_.options());
    for (int axis = 0; axis < dimension_; ++axis) {
      const auto upper = (corner >> axis) & 1;
      offset[static_cast<size_t>(axis + 1)] = upper;
      const auto component = fraction.select(1, axis + 1);
      weight = weight * (upper ? component : 1 - component);
    }
    auto delta = torch::tensor(offset, integer_base.options()).unsqueeze(0);
    lattice_points.push_back(integer_base + delta);
    weight_parts.push_back(std::move(weight));
  }

  auto expanded_coordinates = at::cat(lattice_points, 0);
  auto weights = at::cat(weight_parts, 0);
  auto expanded_features = features_.repeat({corners, 1}) * weights.unsqueeze(1);
  return SparseTensor(std::move(expanded_features),
                      std::move(expanded_coordinates), manager_,
                      SparseTensorQuantizationMode::UNWEIGHTED_SUM);
}

at::Tensor TensorField::inverse_mapping(const CoordinateMapKey& sparse_key) const {
  const auto found = inverse_mappings_.find(sparse_key.get_key());
  if (found != inverse_mappings_.end())
    return found->second;

  auto inserted =
      manager_->field_to_sparse_insert_and_map(field_key_,
                                                sparse_key.get_tensor_stride());
  inverse_mappings_[sparse_key.get_key()] = inserted.second.second;
  return inserted.second.second;
}

at::Tensor TensorField::C() const {
  if (!coordinates_.has_value())
    coordinates_ = manager_->get_coordinates(field_key_);
  return coordinates_.value();
}

at::Tensor TensorField::quantize_features(
    const at::Tensor& first_rows, const at::Tensor& row_to_unique,
    int64_t unique_count, SparseTensorQuantizationMode::Type mode) const {
  switch (mode) {
  case SparseTensorQuantizationMode::RANDOM_SUBSAMPLE:
    return features_.index_select(0, first_rows);
  case SparseTensorQuantizationMode::UNWEIGHTED_SUM:
  case SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE: {
    auto sum = torch::zeros({unique_count, features_.size(1)}, features_.options());
    sum.index_add_(0, row_to_unique, features_);
    if (mode == SparseTensorQuantizationMode::UNWEIGHTED_SUM)
      return sum;
    auto counts = torch::zeros({unique_count}, features_.options());
    counts.index_add_(0, row_to_unique,
                      torch::ones({features_.size(0)}, features_.options()));
    return sum / counts.clamp_min(1).unsqueeze(1);
  }
  case SparseTensorQuantizationMode::MAX_POOL: {
    auto result = torch::zeros({unique_count, features_.size(1)}, features_.options());
    return at::scatter_reduce(
        result, 0, row_to_unique.unsqueeze(1).expand_as(features_), features_,
        "amax", false);
  }
  case SparseTensorQuantizationMode::NO_QUANTIZATION:
    return features_;
  default:
    throw std::runtime_error("Unknown quantization mode");
  }
}

} // namespace riemann
