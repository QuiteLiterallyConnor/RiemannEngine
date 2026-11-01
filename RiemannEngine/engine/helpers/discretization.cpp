#include "helpers/discretization.hpp"
#include "engine/grid/space.hpp"

#include <cmath>
#include <map>

namespace riemann {
namespace {

std::pair<at::Tensor, at::Tensor> insert_map(const at::Tensor& coordinates) {
  TORCH_CHECK(coordinates.dim() == 2 && coordinates.size(1) >= 2,
              "Coordinates must include a batch column and spatial dimensions");
  const int dimension = static_cast<int>(coordinates.size(1) - 1);
  const auto backend = coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                              : CoordinateMapBackend::CPU;
  auto manager = std::make_shared<CoordinateManager>(dimension, backend);
  stride_type stride(static_cast<size_t>(dimension), 1);
  auto inserted = manager->insert_and_map(coordinates.contiguous(), stride);
  return inserted.second;
}

} // namespace

std::vector<at::Tensor> quantize_th(at::Tensor& coordinates) {
  TORCH_CHECK(coordinates.scalar_type() == torch::kInt32,
              "Coordinates must have int32 dtype");
  auto [unique, inverse] = insert_map(coordinates);
  return {unique, inverse};
}

std::vector<at::Tensor> quantize_label_th(at::Tensor coordinates,
                                          at::Tensor labels,
                                          int invalid_label) {
  TORCH_CHECK(coordinates.scalar_type() == torch::kInt32 &&
                  labels.scalar_type() == torch::kInt32,
              "Coordinates and labels must have int32 dtype");
  TORCH_CHECK(coordinates.dim() == 2 && labels.dim() == 1 &&
                  coordinates.size(0) == labels.size(0),
              "Coordinate and label row counts must match");
  TORCH_CHECK(!coordinates.is_cuda() && !labels.is_cuda(),
              "Label quantization currently requires CPU tensors");
  coordinates = coordinates.contiguous();
  labels = labels.contiguous();
  const auto rows = coordinates.size(0);
  const auto columns = coordinates.size(1);
  const auto* coordinate_data = coordinates.data_ptr<int32_t>();
  const auto* label_data = labels.data_ptr<int32_t>();
  std::map<std::vector<int32_t>, int64_t> groups;
  std::vector<int64_t> first_rows;
  std::vector<int64_t> inverse_rows(static_cast<size_t>(rows));
  std::vector<int64_t> coalesced_labels;
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<int32_t> key(
        coordinate_data + row * columns,
        coordinate_data + (row + 1) * columns);
    auto [it, inserted] =
        groups.emplace(std::move(key), static_cast<int64_t>(first_rows.size()));
    const auto group = it->second;
    inverse_rows[static_cast<size_t>(row)] = group;
    if (inserted) {
      first_rows.push_back(row);
      coalesced_labels.push_back(label_data[row]);
    } else if (coalesced_labels[static_cast<size_t>(group)] !=
                   label_data[row] &&
               coalesced_labels[static_cast<size_t>(group)] != invalid_label) {
      coalesced_labels[static_cast<size_t>(group)] = invalid_label;
    }
  }
  auto make_index_tensor = [](const std::vector<int64_t>& values) {
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
  };
  auto unique_rows = make_index_tensor(first_rows);
  auto inverse = make_index_tensor(inverse_rows);
  auto output_labels = make_index_tensor(coalesced_labels);
  return {unique_rows, inverse, output_labels};
}

SparseQuantizeResult sparse_quantize(
    at::Tensor coordinates, c10::optional<at::Tensor> features,
    c10::optional<at::Tensor> labels, int ignore_label,
    c10::optional<double> quantization_size, torch::Device device) {
  TORCH_CHECK(coordinates.dim() == 2,
              "Coordinates must be a 2D matrix");
  TORCH_CHECK(!features.has_value() ||
                  (features->dim() == 2 &&
                   features->size(0) == coordinates.size(0)),
              "Feature rows must match coordinate rows");
  TORCH_CHECK(!labels.has_value() ||
                  (labels->dim() == 1 &&
                   labels->size(0) == coordinates.size(0)),
              "Label rows must match coordinate rows");
  if (quantization_size.has_value())
    TORCH_CHECK(*quantization_size > 0.0 &&
                    std::isfinite(*quantization_size),
                "Quantization size must be positive and finite");

  auto discrete = coordinates.to(torch::kFloat64);
  if (quantization_size.has_value())
    discrete = torch::floor(discrete / *quantization_size);
  else
    discrete = torch::floor(discrete);
  discrete = discrete.to(torch::kInt32).to(device);

  SparseQuantizeResult result;
  if (labels.has_value()) {
    auto cpu_coordinates = discrete.to(torch::kCPU).contiguous();
    auto cpu_labels = labels->to(torch::kInt32).to(torch::kCPU).contiguous();
    auto maps = quantize_label_th(cpu_coordinates, cpu_labels, ignore_label);
    result.unique_index = maps[0];
    result.inverse_mapping = maps[1];
    result.coordinates = cpu_coordinates.index_select(0, maps[0]).to(device);
    result.labels = maps[2];
    if (features.has_value())
      result.features = features->to(device).index_select(0, maps[0].to(device));
  } else {
    auto [unique, inverse] = insert_map(discrete);
    result.unique_index = unique;
    result.inverse_mapping = inverse;
    result.coordinates = discrete.index_select(0, unique.to(torch::kLong));
    if (features.has_value())
      result.features = features->to(device).index_select(
          0, unique.to(features->device(), torch::kLong));
  }
  return result;
}

std::pair<at::Tensor, at::Tensor> unique_coordinate_map(
    at::Tensor coordinates, stride_type tensor_stride) {
  TORCH_CHECK(coordinates.dim() == 2 && coordinates.size(1) >= 2,
              "Coordinates must include a batch column and spatial dimensions");
  const auto dimension = static_cast<size_t>(coordinates.size(1) - 1);
  if (tensor_stride.empty()) tensor_stride.assign(dimension, 1);
  TORCH_CHECK(tensor_stride.size() == dimension,
              "Tensor stride dimension must match coordinates");
  auto manager = std::make_shared<CoordinateManager>(
      static_cast<int>(dimension),
      coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                            : CoordinateMapBackend::CPU);
  return manager->insert_and_map(coordinates.contiguous(), tensor_stride).second;
}

} // namespace riemann
