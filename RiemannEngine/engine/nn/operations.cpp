#include "engine/nn/operations.hpp"

#include <limits>
#include <algorithm>

namespace riemann {
namespace {

void validate_same_map(const std::vector<SparseTensor>& tensors) {
  TORCH_CHECK(tensors.size() > 1,
              "At least two sparse tensors are required");
  const auto manager = tensors.front().coordinate_manager();
  const auto& key = tensors.front().coordinate_map_key();
  for (size_t i = 1; i < tensors.size(); ++i) {
    TORCH_CHECK(manager == tensors[i].coordinate_manager(),
                "Sparse tensors must share a coordinate manager");
    TORCH_CHECK(key == tensors[i].coordinate_map_key(),
                "Sparse tensors must share a coordinate map key");
  }
}

void validate_same_channels(const std::vector<SparseTensor>& tensors) {
  validate_same_map(tensors);
  for (size_t i = 1; i < tensors.size(); ++i)
    TORCH_CHECK(tensors[i].F().size(1) == tensors.front().F().size(1),
                "Sparse tensor feature widths must match");
}

at::Tensor broadcast_global_features(const SparseTensor& input,
                                     const SparseTensor& input_global) {
  TORCH_CHECK(input.F().device() == input_global.F().device(),
              "Broadcast tensors must use the same feature device");
  const auto rows = input.F().size(0);
  const auto global_rows = input_global.F().size(0);
  if (rows == 0)
    return torch::empty({0, input_global.F().size(1)},
                        input_global.F().options());
  TORCH_CHECK(global_rows > 0,
              "Cannot broadcast from an empty global sparse tensor");
  const auto device = input.F().device();
  const auto input_batches =
      input.C().select(1, 0).to(device, torch::kLong);
  const auto global_batches =
      input_global.C().select(1, 0).to(device, torch::kLong);
  const auto matches =
      input_batches.unsqueeze(1).eq(global_batches.unsqueeze(0));
  TORCH_CHECK(matches.sum(1).eq(1).all().item<bool>(),
              "Each input batch must have exactly one global feature row");
  const auto global_indices = matches.to(torch::kLong).argmax(1);
  return input_global.F().index_select(
      0, global_indices.to(input_global.F().device(), torch::kLong));
}

} // namespace

SparseTensor cat(const std::vector<SparseTensor>& tensors) {
  validate_same_map(tensors);
  std::vector<at::Tensor> features;
  features.reserve(tensors.size());
  for (const auto& tensor : tensors) features.push_back(tensor.F());
  return SparseTensor(torch::cat(features, 1),
                      tensors.front().coordinate_map_key(),
                      tensors.front().coordinate_manager());
}

SparseTensor sum(const std::vector<SparseTensor>& tensors) {
  validate_same_channels(tensors);
  auto output = tensors.front().F();
  for (size_t i = 1; i < tensors.size(); ++i)
    output = output + tensors[i].F();
  return SparseTensor(std::move(output), tensors.front().coordinate_map_key(),
                      tensors.front().coordinate_manager());
}

SparseTensor mean(const std::vector<SparseTensor>& tensors) {
  return sum(tensors) * (1.0 / static_cast<double>(tensors.size()));
}

SparseTensor var(const std::vector<SparseTensor>& tensors) {
  validate_same_channels(tensors);
  auto average = sum(tensors).F() / static_cast<double>(tensors.size());
  auto variance = torch::zeros_like(average);
  for (const auto& tensor : tensors)
    variance = variance + (tensor.F() - average).square();
  variance = variance / static_cast<double>(tensors.size());
  return SparseTensor(std::move(variance), tensors.front().coordinate_map_key(),
                      tensors.front().coordinate_manager());
}

at::Tensor dense_coordinates(at::IntArrayRef shape) {
  TORCH_CHECK(shape.size() >= 3,
              "Dense input shape must be [batch, channels, spatial...]");
  int64_t rows = shape[0];
  for (size_t axis = 2; axis < shape.size(); ++axis) {
    TORCH_CHECK(shape[axis] >= 0, "Spatial dimensions cannot be negative");
    TORCH_CHECK(shape[axis] == 0 ||
                    rows <= std::numeric_limits<int64_t>::max() / shape[axis],
                "Dense coordinate count overflows int64");
    rows *= shape[axis];
  }
  const auto dimension = static_cast<int64_t>(shape.size()) - 2;
  auto coords = torch::empty(
      {rows, dimension + 1}, torch::TensorOptions().dtype(torch::kInt32));
  auto values = coords.accessor<int32_t, 2>();
  int64_t row = 0;
  std::vector<int64_t> position(static_cast<size_t>(dimension), 0);
  for (int64_t batch = 0; batch < shape[0]; ++batch) {
    std::fill(position.begin(), position.end(), 0);
    for (int64_t spatial = 0; spatial < rows / std::max<int64_t>(shape[0], 1);
         ++spatial) {
      values[row][0] = static_cast<int32_t>(batch);
      for (int64_t axis = 0; axis < dimension; ++axis)
        values[row][axis + 1] = static_cast<int32_t>(position[axis]);
      for (int64_t axis = dimension - 1; axis >= 0; --axis) {
        if (++position[static_cast<size_t>(axis)] < shape[axis + 2]) break;
        position[static_cast<size_t>(axis)] = 0;
      }
      ++row;
    }
  }
  return coords;
}

namespace {

SparseTensor dense_to_sparse(
    const at::Tensor& dense_tensor,
    const c10::optional<at::Tensor>& coordinates,
    bool remove_zeros) {
  TORCH_CHECK(dense_tensor.dim() >= 3,
              "Dense tensor must have [batch, channels, spatial...] dimensions");
  const auto channels = dense_tensor.size(1);
  auto reduced = dense_tensor.abs().sum(1);
  auto coords = coordinates.has_value()
                    ? coordinates->to(dense_tensor.device())
                    : dense_coordinates(dense_tensor.sizes())
                          .to(dense_tensor.device());
  TORCH_CHECK(coords.dim() == 2 &&
                  coords.size(1) == dense_tensor.dim() - 1,
              "Coordinates must have shape [N, batch+spatial dimension]");
  std::vector<int64_t> permutation{0};
  for (int64_t axis = 2; axis < dense_tensor.dim(); ++axis)
    permutation.push_back(axis);
  permutation.push_back(1);
  auto flat_features =
      dense_tensor.permute(permutation).contiguous().reshape({-1, channels});
  at::Tensor row_indices;
  if (remove_zeros) {
    const auto flat_mask = reduced.reshape({-1}) != 0;
    row_indices = at::nonzero(flat_mask).reshape({-1});
    coords = coords.index_select(0, row_indices.to(torch::kLong));
    flat_features = flat_features.index_select(0, row_indices.to(torch::kLong));
  }
  return SparseTensor(std::move(flat_features), std::move(coords));
}

} // namespace

SparseTensor to_sparse(const at::Tensor& dense_tensor) {
  return dense_to_sparse(dense_tensor, c10::nullopt, true);
}

SparseTensor to_sparse_all(const at::Tensor& dense_tensor,
                           c10::optional<at::Tensor> coordinates) {
  return dense_to_sparse(dense_tensor, coordinates, false);
}

SparseTensor RiemannUnionImpl::forward(
    const std::vector<SparseTensor>& inputs) {
  TORCH_CHECK(inputs.size() > 1,
              "Union requires at least two sparse tensors");
  const auto manager = inputs.front().coordinate_manager();
  std::vector<CoordinateMapKey> keys;
  keys.reserve(inputs.size());
  for (const auto& input : inputs) {
    TORCH_CHECK(input.coordinate_manager() == manager,
                "Union inputs must share a coordinate manager");
    TORCH_CHECK(input.F().size(1) == inputs.front().F().size(1),
                "Union inputs must have equal feature widths");
    keys.push_back(input.coordinate_map_key());
  }
  CoordinateMapKey output_key(inputs.front().coordinate_map_key()
                                   .get_coordinate_size());
  const auto maps = manager->union_map(keys, output_key);
  const auto output_rows = static_cast<int64_t>(manager->size(output_key));
  auto output = torch::zeros(
      {output_rows, inputs.front().F().size(1)}, inputs.front().F().options());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto input_rows = maps[i].index({0}).to(torch::kLong);
    const auto union_rows = maps[i].index({1}).to(torch::kLong);
    output.index_add_(0, union_rows,
                      inputs[i].F().index_select(0, input_rows));
  }
  return SparseTensor(std::move(output), std::move(output_key), manager);
}

SparseTensor RiemannToSparseTensorImpl::forward(const at::Tensor& input) {
  if (remove_zeros_ && !coordinates_.has_value())
    return to_sparse(input);
  return to_sparse_all(input, coordinates_);
}

SparseTensor RiemannPruningImpl::forward(const SparseTensor& input,
                                         const at::Tensor& mask) {
  TORCH_CHECK(mask.numel() == input.F().size(0),
              "Pruning mask must contain one value per sparse row");
  const auto feature_keep = mask.reshape({-1})
                                .to(input.F().device(), torch::kBool)
                                .contiguous();
  const auto coordinates = input.C().index(
      {feature_keep.to(input.C().device(), torch::kBool)});
  const auto filtered_features = input.F().index({feature_keep});
  auto inserted = input.coordinate_manager()->insert_and_map(
      coordinates, input.tensor_stride());
  const auto first_rows =
      inserted.second.first.to(filtered_features.device(), torch::kLong);
  auto features = filtered_features.index_select(0, first_rows);
  return SparseTensor(std::move(features), std::move(inserted.first),
                      input.coordinate_manager());
}

SparseTensor RiemannBroadcastBaseImpl::forward(
    const SparseTensor& input, const SparseTensor& input_global) {
  const auto global_features =
      broadcast_global_features(input, input_global);
  TORCH_CHECK(global_features.size(1) == input.F().size(1) ||
                  global_features.size(1) == 1,
              "Elementwise broadcast requires equal feature widths or a "
              "single global feature channel");
  at::Tensor output;
  switch (operation_type_) {
  case BroadcastMode::ELEMENTWISE_ADDITON:
    output = input.F() + global_features;
    break;
  case BroadcastMode::ELEMENTWISE_MULTIPLICATION:
    output = input.F() * global_features;
    break;
  default:
    TORCH_CHECK(false, "Unsupported sparse broadcast operation");
  }
  return SparseTensor(std::move(output), input.coordinate_map_key(),
                      input.coordinate_manager());
}

SparseTensor RiemannBroadcastAdditionImpl::forward(
    const SparseTensor& input, const SparseTensor& input_global) {
  return RiemannBroadcastBaseImpl(BroadcastMode::ELEMENTWISE_ADDITON)
      .forward(input, input_global);
}

SparseTensor RiemannBroadcastMultiplicationImpl::forward(
    const SparseTensor& input, const SparseTensor& input_global) {
  return RiemannBroadcastBaseImpl(
             BroadcastMode::ELEMENTWISE_MULTIPLICATION)
      .forward(input, input_global);
}

SparseTensor RiemannBroadcastImpl::forward(
    const SparseTensor& input, const SparseTensor& input_global) {
  auto output = broadcast_global_features(input, input_global);
  return SparseTensor(std::move(output), input.coordinate_map_key(),
                      input.coordinate_manager());
}

SparseTensor RiemannBroadcastConcatenationImpl::forward(
    const SparseTensor& input, const SparseTensor& input_global) {
  auto global_features = broadcast_global_features(input, input_global);
  auto output = torch::cat({input.F(), global_features}, 1);
  return SparseTensor(std::move(output), input.coordinate_map_key(),
                      input.coordinate_manager());
}

} // namespace riemann
