#include "engine/tensor/sparse_tensor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef CPU_ONLY
#include <c10/cuda/CUDAGuard.h>
#endif

namespace riemann {
namespace {

int64_t batch_count(const at::Tensor& coordinates) {
  if (coordinates.size(0) == 0)
    return 0;
  return coordinates.select(1, 0).max().item<int64_t>() + 1;
}

at::Tensor align_features(const at::Tensor& features,
                          const at::Tensor& row_to_union,
                          int64_t union_size) {
  auto result = torch::zeros({union_size, features.size(1)}, features.options());
  const auto indices =
      row_to_union.to(features.device(), at::kLong).contiguous();
  result.index_copy_(0, indices, features);
  return result;
}

template <typename Operation>
SparseTensor union_binary_operation(const SparseTensor& lhs,
                                    const SparseTensor& rhs,
                                    Operation operation) {
  const auto coordinates = lhs.C();
  const auto other_coordinates = rhs.C();
  auto union_coordinates = torch::cat({coordinates, other_coordinates}, 0);
  auto output_stride = lhs.tensor_stride();
  const auto other_stride = rhs.tensor_stride();
  for (size_t axis = 0; axis < output_stride.size(); ++axis)
    output_stride[axis] = std::min(output_stride[axis], other_stride[axis]);

  auto output_manager = std::make_shared<CoordinateManager>(
      lhs.dimension(), coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                             : CoordinateMapBackend::CPU);
  auto inserted = output_manager->insert_and_map(
      union_coordinates.contiguous(), std::move(output_stride));
  const int64_t union_size = output_manager->size(inserted.first);
  const auto& row_to_union = inserted.second.second;
  const int64_t lhs_rows = lhs.F().size(0);
  const auto aligned_lhs = align_features(
      lhs.F(), row_to_union.slice(0, 0, lhs_rows), union_size);
  const auto aligned_rhs = align_features(
      rhs.F(), row_to_union.slice(0, lhs_rows, lhs_rows + rhs.F().size(0)),
      union_size);
  return SparseTensor(operation(aligned_lhs, aligned_rhs), inserted.first,
                      std::move(output_manager));
}

} // namespace

SparseTensor::SparseTensor(
    at::Tensor features, at::Tensor coordinates,
    std::shared_ptr<CoordinateManager> manager,
    SparseTensorQuantizationMode::Type quantization_mode,
    GPUMemoryAllocatorBackend::Type alloc_type, RiemannAlgorithm::Mode algo)
    : features_(std::move(features)),
      key_(static_cast<size_type>(coordinates.size(1))),
      dimension_(0) {
  TORCH_CHECK(features_.dim() == 2, "Features must be a 2D tensor");
  TORCH_CHECK(coordinates.dim() == 2, "Coordinates must be a 2D tensor");
  TORCH_CHECK(features_.size(0) == coordinates.size(0),
              "Features and coordinates must have same number of rows");
  TORCH_CHECK(coordinates.size(1) >= 2,
              "Coordinates must contain a batch column and spatial columns");

  dimension_ = static_cast<int>(coordinates.size(1) - 1);
  if (!manager) {
    const auto backend = coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                                : CoordinateMapBackend::CPU;
    manager = std::make_shared<CoordinateManager>(dimension_, backend,
                                                  alloc_type, algo);
  }
  TORCH_CHECK(manager->D() == dimension_,
              "Coordinate manager dimension does not match coordinates");
  TORCH_CHECK(manager->backend() ==
                  (coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                         : CoordinateMapBackend::CPU),
              "Coordinate manager backend does not match coordinate device");
  manager_ = std::move(manager);

  stride_type unit_stride(static_cast<size_t>(dimension_), 1);
  auto inserted = manager_->insert_and_map(coordinates.contiguous(), unit_stride);
  key_ = std::move(inserted.first);
  const auto& first_rows = inserted.second.first;
  const auto& row_to_unique = inserted.second.second;
  features_ = quantize_features(features_, first_rows, row_to_unique,
                                manager_->size(key_), quantization_mode);
}

SparseTensor::SparseTensor(at::Tensor features, CoordinateMapKey coordinate_map_key,
                           std::shared_ptr<CoordinateManager> manager)
    : features_(std::move(features)),
      key_(std::move(coordinate_map_key)),
      manager_(std::move(manager)),
      dimension_(static_cast<int>(key_.get_coordinate_size()) - 1) {
  TORCH_CHECK(manager_, "Manager must not be null");
  TORCH_CHECK(features_.dim() == 2, "Features must be 2D");
  TORCH_CHECK(key_.is_key_set(), "Coordinate map key must be set");
  TORCH_CHECK(manager_->D() == dimension_,
              "Coordinate manager dimension does not match coordinate key");
  TORCH_CHECK(features_.size(0) == manager_->size(key_),
              "Feature row count must match coordinate map size");
}

at::Tensor SparseTensor::C() const {
  if (!coordinates_.has_value())
    coordinates_ = manager_->get_coordinates(key_);
  return coordinates_.value();
}

stride_type SparseTensor::tensor_stride() const {
  return key_.get_tensor_stride();
}

std::vector<at::Tensor> SparseTensor::decomposed_coordinates() const {
  const auto coordinates = C();
  const auto count = batch_count(coordinates);
  std::vector<at::Tensor> result;
  result.reserve(static_cast<size_t>(count));
  for (int64_t batch = 0; batch < count; ++batch)
    result.push_back(coordinates.index(
        {coordinates.select(1, 0) == batch}));
  return result;
}

std::vector<at::Tensor> SparseTensor::decomposed_features() const {
  const auto coordinates = C();
  const auto count = batch_count(coordinates);
  std::vector<at::Tensor> result;
  result.reserve(static_cast<size_t>(count));
  for (int64_t batch = 0; batch < count; ++batch)
    result.push_back(features_.index({coordinates.select(1, 0) == batch}));
  return result;
}

std::vector<std::pair<at::Tensor, at::Tensor>>
SparseTensor::decomposed_coordinates_and_features() const {
  const auto coordinates = C();
  const auto count = batch_count(coordinates);
  std::vector<std::pair<at::Tensor, at::Tensor>> result;
  result.reserve(static_cast<size_t>(count));
  for (int64_t batch = 0; batch < count; ++batch) {
    const auto mask = coordinates.select(1, 0) == batch;
    result.emplace_back(coordinates.index({mask}), features_.index({mask}));
  }
  return result;
}

at::Tensor SparseTensor::dense(
    c10::optional<std::vector<int64_t>> shape,
    c10::optional<at::Tensor> min_coordinate) const {
  const auto coordinates = C();
  TORCH_CHECK(features_.device() == coordinates.device(),
              "Features and coordinates must be on the same device");
  TORCH_CHECK(coordinates.dim() == 2 &&
                  coordinates.size(1) == dimension_ + 1,
              "Coordinate tensor shape does not match sparse tensor dimension");

  const int64_t channels = features_.size(1);
  const int64_t coordinate_count = coordinates.size(0);
  int64_t batch_origin = 0;
  std::vector<int64_t> spatial_origin(static_cast<size_t>(dimension_), 0);

  if (min_coordinate.has_value()) {
    const auto& minimum = min_coordinate.value();
    TORCH_CHECK(minimum.numel() == dimension_ ||
                    minimum.numel() == dimension_ + 1,
                "min_coordinate must contain D spatial values or a batch "
                "value followed by D spatial values");
    TORCH_CHECK(!minimum.is_floating_point() && !minimum.is_complex() &&
                    minimum.scalar_type() != at::kBool,
                "min_coordinate must have an integral dtype");
    const auto minimum_values =
        minimum.reshape({-1}).to(torch::kCPU).to(at::kLong).contiguous();
    const auto* values = minimum_values.data_ptr<int64_t>();
    const int64_t offset = minimum.numel() == dimension_ + 1 ? 1 : 0;
    if (offset != 0)
      batch_origin = values[0];
    for (int64_t axis = 0; axis < dimension_; ++axis)
      spatial_origin[static_cast<size_t>(axis)] = values[axis + offset];
  } else if (coordinate_count != 0) {
    for (int64_t axis = 0; axis < dimension_; ++axis)
      spatial_origin[static_cast<size_t>(axis)] =
          coordinates.select(1, axis + 1).min().item<int64_t>();
  }
  TORCH_CHECK(
      batch_origin >= std::numeric_limits<int32_t>::min() &&
          batch_origin <= std::numeric_limits<int32_t>::max(),
      "min_coordinate batch value is outside the coordinate range");
  for (const int64_t origin : spatial_origin)
    TORCH_CHECK(origin >= std::numeric_limits<int32_t>::min() &&
                    origin <= std::numeric_limits<int32_t>::max(),
                "min_coordinate spatial value is outside the coordinate range");

  std::vector<int64_t> spatial_shape(static_cast<size_t>(dimension_), 0);
  int64_t batch_size = 0;
  if (coordinate_count != 0) {
    const auto batches = coordinates.select(1, 0);
    const int64_t max_batch = batches.max().item<int64_t>();
    TORCH_CHECK(max_batch >= batch_origin,
                "min_coordinate places all batch coordinates beyond the "
                "dense tensor");
    batch_size = max_batch - batch_origin + 1;
    for (int64_t axis = 0; axis < dimension_; ++axis) {
      const int64_t max_coordinate =
          coordinates.select(1, axis + 1).max().item<int64_t>();
      const int64_t origin = spatial_origin[static_cast<size_t>(axis)];
      TORCH_CHECK(max_coordinate >= origin,
                  "min_coordinate places spatial coordinates beyond the "
                  "dense tensor");
      spatial_shape[static_cast<size_t>(axis)] = max_coordinate - origin + 1;
    }
  }

  if (shape.has_value()) {
    const auto& requested = shape.value();
    const bool includes_batch =
        requested.size() == static_cast<size_t>(dimension_ + 1) ||
        requested.size() == static_cast<size_t>(dimension_ + 2);
    const bool includes_channels =
        requested.size() == static_cast<size_t>(dimension_ + 2);
    TORCH_CHECK(requested.size() == static_cast<size_t>(dimension_) ||
                    includes_batch,
                "shape must contain spatial extents, batch plus spatial "
                "extents, or the complete dense shape");

    size_t spatial_offset = 0;
    if (includes_batch) {
      TORCH_CHECK(requested[0] >= 0, "Dense batch size must be non-negative");
      batch_size = requested[0];
      spatial_offset = 1;
    }
    if (includes_channels) {
      TORCH_CHECK(requested[1] == channels,
                  "Dense shape channel count must match feature channels");
      spatial_offset = 2;
    }
    for (int64_t axis = 0; axis < dimension_; ++axis) {
      const int64_t extent =
          requested[spatial_offset + static_cast<size_t>(axis)];
      TORCH_CHECK(extent >= 0,
                  "Dense spatial extents must be non-negative");
      spatial_shape[static_cast<size_t>(axis)] = extent;
    }
  }

  auto coordinate_indices = coordinates.to(at::kLong);
  auto long_options =
      coordinates.options().dtype(at::kLong).requires_grad(false);
  auto batch_indices =
      coordinate_indices.select(1, 0).sub(batch_origin);
  TORCH_CHECK(coordinate_count == 0 ||
                  (batch_indices.min().item<int64_t>() >= 0 &&
                   batch_indices.max().item<int64_t>() < batch_size),
              "Dense batch shape does not contain all coordinates");

  int64_t spatial_volume = 1;
  auto spatial_indices = torch::zeros({coordinate_count}, long_options);
  for (int64_t axis = 0; axis < dimension_; ++axis) {
    const int64_t extent = spatial_shape[static_cast<size_t>(axis)];
    TORCH_CHECK(extent == 0 ||
                    spatial_volume <=
                        std::numeric_limits<int64_t>::max() / extent,
                "Dense spatial shape is too large");
    spatial_volume *= extent;
    const int64_t origin = spatial_origin[static_cast<size_t>(axis)];
    const auto axis_indices =
        coordinate_indices.select(1, axis + 1).sub(origin);
    TORCH_CHECK(coordinate_count == 0 ||
                    (axis_indices.min().item<int64_t>() >= 0 &&
                     axis_indices.max().item<int64_t>() < extent),
                "Dense spatial shape does not contain all coordinates");
    spatial_indices = spatial_indices * extent + axis_indices;
  }

  TORCH_CHECK(channels == 0 ||
                  batch_size <= std::numeric_limits<int64_t>::max() / channels,
              "Dense shape is too large");
  const int64_t batch_channels = batch_size * channels;
  TORCH_CHECK(spatial_volume == 0 ||
                  batch_channels <=
                      std::numeric_limits<int64_t>::max() / spatial_volume,
              "Dense shape is too large");
  const int64_t dense_size = batch_channels * spatial_volume;
  auto flat = torch::zeros({dense_size}, features_.options());
  if (coordinate_count != 0 && channels != 0) {
    const auto point_offsets =
        (batch_indices * channels) * spatial_volume + spatial_indices;
    const auto channel_offsets =
        torch::arange(channels, long_options).unsqueeze(0) * spatial_volume;
    const auto destination =
        (point_offsets.unsqueeze(1) + channel_offsets).reshape({-1});
    flat = torch::scatter(flat, 0, destination, features_.reshape({-1}));
  }

  std::vector<int64_t> dense_shape;
  dense_shape.reserve(static_cast<size_t>(dimension_ + 2));
  dense_shape.push_back(batch_size);
  dense_shape.push_back(channels);
  dense_shape.insert(dense_shape.end(), spatial_shape.begin(),
                     spatial_shape.end());
  return flat.reshape(dense_shape);
}

at::Tensor SparseTensor::sparse() const {
  const auto coordinates = C();
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(dimension_ + 2));
  for (int64_t axis = 0; axis <= dimension_; ++axis) {
    TORCH_CHECK(coordinates.size(0) != 0,
                "Cannot infer sparse tensor shape from empty coordinates");
    shape.push_back(
        coordinates.select(1, axis).max().item<int64_t>() + 1);
  }
  shape.push_back(features_.size(1));
  return torch::sparse_coo_tensor(coordinates.transpose(0, 1).to(torch::kLong),
                                  features_, shape);
}

void SparseTensor::require_compatible(const SparseTensor& other,
                                      const char* operation) const {
  TORCH_CHECK(dimension_ == other.dimension_,
              "Cannot ", operation, " tensors with different dimensions");
  TORCH_CHECK(features_.size(1) == other.features_.size(1),
              "Cannot ", operation,
              " tensors with different feature channel counts");
  TORCH_CHECK(features_.device() == other.features_.device(),
              "Cannot ", operation, " tensors on different devices");
  TORCH_CHECK(features_.device() == C().device() &&
                  other.features_.device() == other.C().device(),
              "Features and coordinates must be on the same device");
}

SparseTensor SparseTensor::operator+(const SparseTensor& other) const {
  require_compatible(other, "add");
  return union_binary_operation(*this, other,
                                [](const at::Tensor& lhs,
                                   const at::Tensor& rhs) {
                                  return lhs + rhs;
                                });
}

SparseTensor SparseTensor::operator-(const SparseTensor& other) const {
  require_compatible(other, "subtract");
  return union_binary_operation(*this, other,
                                [](const at::Tensor& lhs,
                                   const at::Tensor& rhs) {
                                  return lhs - rhs;
                                });
}

SparseTensor SparseTensor::operator*(const SparseTensor& other) const {
  require_compatible(other, "multiply");
  return union_binary_operation(*this, other,
                                [](const at::Tensor& lhs,
                                   const at::Tensor& rhs) {
                                  return lhs * rhs;
                                });
}

SparseTensor SparseTensor::operator/(const SparseTensor& other) const {
  require_compatible(other, "divide");
  return union_binary_operation(*this, other,
                                [](const at::Tensor& lhs,
                                   const at::Tensor& rhs) {
                                  return lhs / rhs;
                                });
}

SparseTensor& SparseTensor::operator+=(const SparseTensor& other) {
  TORCH_CHECK(manager_ == other.manager_,
              "Cannot add tensors from different managers");
  TORCH_CHECK(key_ == other.key_, "In-place addition requires same key");
  features_ += other.features_;
  return *this;
}

SparseTensor SparseTensor::operator+(at::Scalar scalar) const {
  return SparseTensor(features_ + scalar, key_, manager_);
}

SparseTensor SparseTensor::operator*(at::Scalar scalar) const {
  return SparseTensor(features_ * scalar, key_, manager_);
}

SparseTensor SparseTensor::to(torch::Device device) const {
  TORCH_CHECK(device.type() == torch::kCPU || device.type() == torch::kCUDA,
              "SparseTensor supports transfer only to CPU or CUDA devices");
#ifdef CPU_ONLY
  TORCH_CHECK(device.type() == torch::kCPU,
              "CUDA device transfer is not available in a CPU_ONLY build");
#endif

#ifndef CPU_ONLY
  std::unique_ptr<c10::cuda::CUDAGuard> device_guard;
  if (device.type() == torch::kCUDA)
    device_guard = std::make_unique<c10::cuda::CUDAGuard>(device);
#endif

  const auto coordinates = C();
  TORCH_CHECK(features_.device() == coordinates.device(),
              "Features and coordinates must be on the same device");
  auto moved_features = features_.to(device);
  auto moved_coordinates = coordinates.to(device).contiguous();
  auto output_manager = std::make_shared<CoordinateManager>(
      dimension_, moved_coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                             : CoordinateMapBackend::CPU);
  auto inserted = output_manager->insert_and_map(
      moved_coordinates, key_.get_tensor_stride(), key_.get_key().second);
  const auto& first_rows = inserted.second.first;
  const int64_t output_rows = output_manager->size(inserted.first);
  TORCH_CHECK(first_rows.numel() == output_rows,
              "Coordinate map representative rows do not match map size");
  auto moved_aligned = moved_features.index_select(
      0, first_rows.to(device, at::kLong).contiguous());
  return SparseTensor(std::move(moved_aligned), std::move(inserted.first),
                      std::move(output_manager));
}

SparseTensor SparseTensor::to(c10::ScalarType dtype) const {
  return SparseTensor(features_.to(dtype), key_, manager_);
}

SparseTensor SparseTensor::float_() const {
  return to(at::kFloat);
}

SparseTensor SparseTensor::double_() const {
  return to(at::kDouble);
}

SparseTensor SparseTensor::detach() const {
  return SparseTensor(features_.detach(), key_, manager_);
}

at::Tensor SparseTensor::quantize_features(
    const at::Tensor& features, const at::Tensor& first_rows,
    const at::Tensor& row_to_unique, int64_t unique_count,
    SparseTensorQuantizationMode::Type mode) const {
  switch (mode) {
    case SparseTensorQuantizationMode::RANDOM_SUBSAMPLE: {
      if (features.size(0) == unique_count &&
          at::equal(first_rows,
                    torch::arange(features.size(0), first_rows.options())))
        return features;
      return features.index_select(0, first_rows);
    }
  case SparseTensorQuantizationMode::NO_QUANTIZATION:
    return features;
  case SparseTensorQuantizationMode::UNWEIGHTED_SUM:
  case SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE: {
    auto result = torch::zeros({unique_count, features.size(1)}, features.options());
    result.index_add_(0, row_to_unique, features);
    if (mode == SparseTensorQuantizationMode::UNWEIGHTED_SUM)
      return result;
    auto counts = torch::zeros({unique_count}, features.options());
    counts.index_add_(0, row_to_unique,
                      torch::ones({features.size(0)}, features.options()));
    return result / counts.clamp_min(1).unsqueeze(1);
  }
  case SparseTensorQuantizationMode::MAX_POOL: {
    auto result = torch::zeros({unique_count, features.size(1)}, features.options());
    return at::scatter_reduce(
        result, 0, row_to_unique.unsqueeze(1).expand_as(features), features,
        "amax", false);
  }
  default:
    throw std::runtime_error("Unknown quantization mode");
  }
}

} // namespace riemann
