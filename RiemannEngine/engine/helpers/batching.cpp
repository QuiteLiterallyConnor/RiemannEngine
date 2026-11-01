#include "helpers/batching.hpp"

namespace riemann {
namespace {

torch::Device resolve_device(const std::vector<at::Tensor>& coordinates,
                             const c10::optional<torch::Device>& requested) {
  if (requested.has_value()) return *requested;
  return coordinates.empty() ? torch::Device(torch::kCPU)
                             : coordinates.front().device();
}

at::Tensor normalize_coordinates(const at::Tensor& coordinates,
                                 c10::ScalarType dtype,
                                 const torch::Device& device) {
  auto result = coordinates;
  if (dtype == torch::kInt32 &&
      (result.scalar_type() == torch::kFloat32 ||
       result.scalar_type() == torch::kFloat64))
    result = result.floor();
  return result.to(device, dtype);
}

} // namespace

at::Tensor batched_coordinates(const std::vector<at::Tensor>& coordinates,
                               c10::ScalarType dtype,
                               c10::optional<torch::Device> device) {
  TORCH_CHECK(!coordinates.empty(), "Coordinates sequence must not be empty");
  TORCH_CHECK(dtype == torch::kInt32 || dtype == torch::kFloat32,
              "Coordinate dtype must be int32 or float32");
  const auto dimension = coordinates.front().dim() == 2
                             ? coordinates.front().size(1)
                             : -1;
  int64_t rows = 0;
  for (size_t i = 0; i < coordinates.size(); ++i) {
    TORCH_CHECK(coordinates[i].dim() == 2 &&
                    coordinates[i].size(1) == dimension,
                "Coordinate tensor ", i, " must be a matrix with ", dimension,
                " columns");
    rows += coordinates[i].size(0);
  }
  const auto target = resolve_device(coordinates, device);
  auto output = torch::empty(
      {rows, dimension + 1},
      torch::TensorOptions().dtype(dtype).device(target));
  int64_t offset = 0;
  for (size_t batch = 0; batch < coordinates.size(); ++batch) {
    const auto count = coordinates[batch].size(0);
    auto slice = output.narrow(0, offset, count);
    slice.select(1, 0).fill_(static_cast<int64_t>(batch));
    slice.narrow(1, 1, dimension).copy_(
        normalize_coordinates(coordinates[batch], dtype, target));
    offset += count;
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sparse_collate(const std::vector<at::Tensor>& coordinates,
               const std::vector<at::Tensor>& features,
               const std::vector<at::Tensor>& labels,
               c10::ScalarType dtype,
               c10::optional<torch::Device> device) {
  TORCH_CHECK(!coordinates.empty(), "Coordinates must not be empty");
  TORCH_CHECK(coordinates.size() == features.size(),
              "Coordinate and feature sample counts must match");
  TORCH_CHECK(labels.empty() || labels.size() == coordinates.size(),
              "Label sample count must match coordinate sample count");
  const auto target = resolve_device(coordinates, device);
  const auto dims = coordinates.front().dim() == 2
                        ? coordinates.front().size(1)
                        : -1;
  std::vector<at::Tensor> moved_features, moved_labels;
  moved_features.reserve(features.size());
  moved_labels.reserve(labels.size());
  for (size_t i = 0; i < coordinates.size(); ++i) {
    TORCH_CHECK(coordinates[i].dim() == 2 && features[i].dim() == 2 &&
                    coordinates[i].size(1) == dims &&
                    coordinates[i].size(0) == features[i].size(0),
                "Invalid coordinate/feature shapes for sample ", i);
    TORCH_CHECK(labels.empty() || labels[i].size(0) == coordinates[i].size(0),
                "Coordinate and label counts must match for sample ", i);
    moved_features.push_back(features[i].to(target, torch::kFloat));
    if (!labels.empty()) moved_labels.push_back(labels[i].to(target));
  }

  auto batch_coords = batched_coordinates(coordinates, dtype, target);
  auto batch_features = torch::cat(moved_features, 0);
  if (labels.empty())
    return {std::move(batch_coords), std::move(batch_features), c10::nullopt};
  return {std::move(batch_coords), std::move(batch_features),
          torch::cat(moved_labels, 0)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_sparse_collate(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& data,
    c10::ScalarType dtype, c10::optional<torch::Device> device) {
  std::vector<at::Tensor> coordinates, features, labels;
  coordinates.reserve(data.size());
  features.reserve(data.size());
  labels.reserve(data.size());
  for (const auto& sample : data) {
    coordinates.push_back(std::get<0>(sample));
    features.push_back(std::get<1>(sample));
    labels.push_back(std::get<2>(sample));
  }
  auto [batch_coords, batch_features, batch_labels] =
      sparse_collate(coordinates, features, labels, dtype, device);
  TORCH_CHECK(batch_labels.has_value(), "Expected labels in sparse batch");
  return {std::move(batch_coords), std::move(batch_features), *batch_labels};
}

SparseCollation::SparseCollation(int64_t limit_numpoints, c10::ScalarType dtype,
                                 c10::optional<torch::Device> device)
    : limit_numpoints_(limit_numpoints), dtype_(dtype), device_(device) {
  TORCH_CHECK(limit_numpoints_ == -1 || limit_numpoints_ > 0,
              "Point limit must be -1 or positive");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
SparseCollation::operator()(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& data) const {
  TORCH_CHECK(!data.empty(), "Cannot collate an empty sample list");
  std::vector<at::Tensor> coordinates, features, labels;
  int64_t count = 0;
  for (const auto& sample : data) {
    const auto sample_count = std::get<0>(sample).size(0);
    if (limit_numpoints_ > 0 && count + sample_count > limit_numpoints_) break;
    count += sample_count;
    coordinates.push_back(std::get<0>(sample));
    features.push_back(std::get<1>(sample));
    labels.push_back(std::get<2>(sample));
  }
  TORCH_CHECK(!coordinates.empty(),
              "No sample fits within the configured point limit");
  auto [batch_coords, batch_features, batch_labels] =
      sparse_collate(coordinates, features, labels, dtype_, device_);
  TORCH_CHECK(batch_labels.has_value(), "Expected labels in sparse batch");
  return {std::move(batch_coords), std::move(batch_features), *batch_labels};
}

} // namespace riemann
