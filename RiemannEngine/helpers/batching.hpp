#ifndef RIEMANN_ENGINE_PUBLIC_BATCHING_HPP
#define RIEMANN_ENGINE_PUBLIC_BATCHING_HPP

#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace riemann {

at::Tensor batched_coordinates(
    const std::vector<at::Tensor>& coordinates,
    c10::ScalarType dtype = torch::kInt32,
    c10::optional<torch::Device> device = c10::nullopt);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> sparse_collate(
    const std::vector<at::Tensor>& coordinates,
    const std::vector<at::Tensor>& features,
    const std::vector<at::Tensor>& labels = {},
    c10::ScalarType dtype = torch::kInt32,
    c10::optional<torch::Device> device = c10::nullopt);

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_sparse_collate(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& data,
    c10::ScalarType dtype = torch::kInt32,
    c10::optional<torch::Device> device = c10::nullopt);

class SparseCollation {
public:
  explicit SparseCollation(int64_t limit_numpoints = -1,
                           c10::ScalarType dtype = torch::kInt32,
                           c10::optional<torch::Device> device = c10::nullopt);
  std::tuple<at::Tensor, at::Tensor, at::Tensor> operator()(
      const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& data) const;

private:
  int64_t limit_numpoints_;
  c10::ScalarType dtype_;
  c10::optional<torch::Device> device_;
};

} // namespace riemann

#endif
