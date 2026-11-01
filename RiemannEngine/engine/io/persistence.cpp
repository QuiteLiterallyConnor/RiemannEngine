#include "persistence.hpp"

#include <torch/serialize.h>

#include <cstring>
#include <limits>

namespace riemann {
namespace {

at::Tensor make_int64_vector(const std::vector<int64_t>& values) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}

std::vector<int64_t> read_int64_vector(const at::Tensor& tensor) {
  auto values = tensor.to(torch::kCPU, torch::kInt64).contiguous();
  const auto* begin = values.data_ptr<int64_t>();
  return std::vector<int64_t>(begin, begin + values.numel());
}

std::vector<int64_t> widen_stride(const stride_type& stride) {
  return std::vector<int64_t>(stride.begin(), stride.end());
}

void write_sparse(torch::serialize::OutputArchive& archive,
                  const std::string& prefix, const SparseTensorData& data) {
  archive.write(prefix + "coordinates", data.coordinates.contiguous());
  archive.write(prefix + "features", data.features.contiguous());
  archive.write(prefix + "dimension",
                torch::tensor(data.dimension, torch::kInt64));
  archive.write(prefix + "tensor_stride", make_int64_vector(data.tensor_stride));
}

SparseTensorData read_sparse(torch::serialize::InputArchive& archive,
                             const std::string& prefix) {
  SparseTensorData data;
  archive.read(prefix + "coordinates", data.coordinates);
  archive.read(prefix + "features", data.features);
  at::Tensor dimension;
  at::Tensor stride;
  archive.read(prefix + "dimension", dimension);
  archive.read(prefix + "tensor_stride", stride);
  data.dimension = static_cast<int>(dimension.item<int64_t>());
  data.tensor_stride = read_int64_vector(stride);
  TORCH_CHECK(data.coordinates.dim() == 2 && data.features.dim() == 2 &&
                  data.coordinates.size(0) == data.features.size(0) &&
                  data.coordinates.size(1) == data.dimension + 1 &&
                  data.tensor_stride.size() ==
                      static_cast<size_t>(data.dimension),
              "Invalid sparse tensor data in archive");
  return data;
}

} // namespace

void save_tensors(
    const std::vector<std::pair<std::string, at::Tensor>>& tensors,
    const std::string& path) {
  torch::serialize::OutputArchive archive;
  archive.write("count", torch::tensor(
                             static_cast<int64_t>(tensors.size()),
                             torch::TensorOptions().dtype(torch::kInt64)));
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto index = std::to_string(i);
    const auto& [name, tensor] = tensors[i];
    auto encoded_name = torch::empty(
        {static_cast<int64_t>(name.size())},
        torch::TensorOptions().dtype(torch::kUInt8));
    if (!name.empty())
      std::memcpy(encoded_name.data_ptr<uint8_t>(), name.data(), name.size());
    archive.write("name_" + index, encoded_name);
    archive.write("tensor_" + index, tensor);
  }
  archive.save_to(path);
}

std::vector<std::pair<std::string, at::Tensor>>
load_tensors(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  at::Tensor count_tensor;
  archive.read("count", count_tensor);
  const auto count = count_tensor.item<int64_t>();
  TORCH_CHECK(count >= 0, "Invalid tensor count in archive");
  std::vector<std::pair<std::string, at::Tensor>> tensors;
  tensors.reserve(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const auto index = std::to_string(i);
    at::Tensor encoded_name;
    at::Tensor tensor;
    archive.read("name_" + index, encoded_name);
    archive.read("tensor_" + index, tensor);
    encoded_name = encoded_name.to(torch::kCPU, torch::kUInt8).contiguous();
    const auto* bytes = reinterpret_cast<const char*>(
        encoded_name.data_ptr<uint8_t>());
    tensors.emplace_back(std::string(bytes, static_cast<size_t>(encoded_name.numel())),
                         std::move(tensor));
  }
  return tensors;
}

void save_sparse_tensor(const SparseTensor& tensor, const std::string& path) {
  SparseTensorData data{tensor.C(), tensor.F(),
                        widen_stride(tensor.tensor_stride()),
                        tensor.dimension()};
  torch::serialize::OutputArchive archive;
  write_sparse(archive, "", data);
  archive.save_to(path);
}

SparseTensorData load_sparse_tensor_data(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  return read_sparse(archive, "");
}

SparseTensor reconstruct_sparse_tensor(
    const SparseTensorData& data,
    std::shared_ptr<CoordinateManager> manager) {
  TORCH_CHECK(data.dimension > 0 &&
                  data.coordinates.size(1) == data.dimension + 1 &&
                  data.tensor_stride.size() ==
                      static_cast<size_t>(data.dimension),
              "Sparse tensor dimension and stride are inconsistent");
  if (!manager) {
    manager = std::make_shared<CoordinateManager>(
        data.dimension,
        data.coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                   : CoordinateMapBackend::CPU);
  }
  TORCH_CHECK(manager->D() == data.dimension,
              "Coordinate manager dimension mismatch");
  stride_type stride;
  stride.reserve(data.tensor_stride.size());
  for (const auto value : data.tensor_stride) {
    TORCH_CHECK(value > 0 &&
                    value <= std::numeric_limits<size_type>::max(),
                "Invalid tensor stride in sparse tensor archive");
    stride.push_back(static_cast<size_type>(value));
  }
  auto inserted =
      manager->insert_and_map(data.coordinates.contiguous(), std::move(stride));
  auto reordered = data.features.index_select(
      0, inserted.second.first.to(data.features.device(), torch::kLong));
  return SparseTensor(std::move(reordered), std::move(inserted.first),
                      std::move(manager));
}

SparseTensor load_sparse_tensor(const std::string& path) {
  return reconstruct_sparse_tensor(load_sparse_tensor_data(path));
}

void save_sparse_tensors(const std::vector<SparseTensor>& tensors,
                         const std::string& path) {
  torch::serialize::OutputArchive archive;
  archive.write("count", torch::tensor(
                             static_cast<int64_t>(tensors.size()),
                             torch::TensorOptions().dtype(torch::kInt64)));
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto prefix = "st_" + std::to_string(i) + "_";
    SparseTensorData data{tensors[i].C(), tensors[i].F(),
                          widen_stride(tensors[i].tensor_stride()),
                          tensors[i].dimension()};
    write_sparse(archive, prefix, data);
  }
  archive.save_to(path);
}

std::vector<SparseTensorData> load_sparse_tensors_data(
    const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  at::Tensor count_tensor;
  archive.read("count", count_tensor);
  const auto count = count_tensor.item<int64_t>();
  TORCH_CHECK(count >= 0, "Invalid sparse tensor count in archive");
  std::vector<SparseTensorData> tensors;
  tensors.reserve(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i)
    tensors.push_back(read_sparse(archive, "st_" + std::to_string(i) + "_"));
  return tensors;
}

} // namespace riemann
