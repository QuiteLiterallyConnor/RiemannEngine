#ifndef RIEMANN_ENGINE_NN_OPERATIONS_HPP
#define RIEMANN_ENGINE_NN_OPERATIONS_HPP

#include "engine/tensor/sparse_tensor.hpp"

#include <torch/torch.h>

namespace riemann {

SparseTensor cat(const std::vector<SparseTensor>& tensors);
SparseTensor sum(const std::vector<SparseTensor>& tensors);
SparseTensor mean(const std::vector<SparseTensor>& tensors);
SparseTensor var(const std::vector<SparseTensor>& tensors);
at::Tensor dense_coordinates(at::IntArrayRef shape);
SparseTensor to_sparse(const at::Tensor& dense_tensor);
SparseTensor to_sparse_all(
    const at::Tensor& dense_tensor,
    c10::optional<at::Tensor> coordinates = c10::nullopt);

class RiemannUnionImpl : public torch::nn::Module {
public:
  SparseTensor forward(const std::vector<SparseTensor>& inputs);
};
TORCH_MODULE(RiemannUnion);

class RiemannToSparseTensorImpl : public torch::nn::Module {
public:
  explicit RiemannToSparseTensorImpl(
      bool remove_zeros = true,
      c10::optional<at::Tensor> coordinates = c10::nullopt)
      : remove_zeros_(remove_zeros), coordinates_(std::move(coordinates)) {}
  SparseTensor forward(const at::Tensor& input);

private:
  bool remove_zeros_;
  c10::optional<at::Tensor> coordinates_;
};
TORCH_MODULE(RiemannToSparseTensor);

class RiemannToDenseTensorImpl : public torch::nn::Module {
public:
  explicit RiemannToDenseTensorImpl(
      c10::optional<std::vector<int64_t>> shape = c10::nullopt)
      : shape_(std::move(shape)) {}
  at::Tensor forward(const SparseTensor& input) { return input.dense(shape_); }

private:
  c10::optional<std::vector<int64_t>> shape_;
};
TORCH_MODULE(RiemannToDenseTensor);

class RiemannToFeatureImpl : public torch::nn::Module {
public:
  at::Tensor forward(const SparseTensor& input) { return input.F(); }
};
TORCH_MODULE(RiemannToFeature);

class RiemannPruningImpl : public torch::nn::Module {
public:
  SparseTensor forward(const SparseTensor& input, const at::Tensor& mask);
};
TORCH_MODULE(RiemannPruning);

class RiemannBroadcastBaseImpl : public torch::nn::Module {
public:
  explicit RiemannBroadcastBaseImpl(BroadcastMode::Type operation_type)
      : operation_type_(operation_type) {}
  SparseTensor forward(const SparseTensor& input,
                       const SparseTensor& input_global);

private:
  BroadcastMode::Type operation_type_;
};

class RiemannBroadcastAdditionImpl : public torch::nn::Module {
public:
  SparseTensor forward(const SparseTensor& input,
                       const SparseTensor& input_global);
};
TORCH_MODULE(RiemannBroadcastAddition);

class RiemannBroadcastMultiplicationImpl : public torch::nn::Module {
public:
  SparseTensor forward(const SparseTensor& input,
                       const SparseTensor& input_global);
};
TORCH_MODULE(RiemannBroadcastMultiplication);

class RiemannBroadcastImpl : public torch::nn::Module {
public:
  SparseTensor forward(const SparseTensor& input,
                       const SparseTensor& input_global);
};
TORCH_MODULE(RiemannBroadcast);

class RiemannBroadcastConcatenationImpl : public torch::nn::Module {
public:
  SparseTensor forward(const SparseTensor& input,
                       const SparseTensor& input_global);
};
TORCH_MODULE(RiemannBroadcastConcatenation);

} // namespace riemann

#endif
