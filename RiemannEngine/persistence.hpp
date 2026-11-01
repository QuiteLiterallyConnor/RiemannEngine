#ifndef RIEMANN_ENGINE_PUBLIC_PERSISTENCE_HPP
#define RIEMANN_ENGINE_PUBLIC_PERSISTENCE_HPP

#include <torch/torch.h>

#include "engine/tensor/sparse_tensor.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace riemann {

template <typename Module>
void save_model(const Module& module, const std::string& path) {
  torch::save(module, path);
}

template <typename Module>
void load_model(Module& module, const std::string& path) {
  torch::load(module, path);
}

template <typename ModuleImpl>
void save_model(const std::shared_ptr<ModuleImpl>& module,
                const std::string& path) {
  torch::save(module, path);
}

template <typename ModuleImpl>
void load_model(std::shared_ptr<ModuleImpl>& module,
                const std::string& path) {
  torch::load(module, path);
}

void save_tensors(
    const std::vector<std::pair<std::string, at::Tensor>>& tensors,
    const std::string& path);
std::vector<std::pair<std::string, at::Tensor>>
load_tensors(const std::string& path);

struct SparseTensorData {
  at::Tensor coordinates;
  at::Tensor features;
  std::vector<int64_t> tensor_stride;
  int dimension = 0;
};

void save_sparse_tensor(const SparseTensor& tensor, const std::string& path);
SparseTensorData load_sparse_tensor_data(const std::string& path);
SparseTensor reconstruct_sparse_tensor(
    const SparseTensorData& data,
    std::shared_ptr<CoordinateManager> manager = nullptr);
SparseTensor load_sparse_tensor(const std::string& path);
void save_sparse_tensors(const std::vector<SparseTensor>& tensors,
                         const std::string& path);
std::vector<SparseTensorData> load_sparse_tensors_data(const std::string& path);

} // namespace riemann

#endif
