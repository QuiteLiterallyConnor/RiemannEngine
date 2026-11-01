#include "engine/nn/resample.hpp"

namespace riemann {
namespace {

at::Tensor concatenate_or_empty(const std::vector<at::Tensor>& tensors,
                                const at::TensorOptions& options) {
  if (tensors.empty()) return torch::empty({0}, options);
  return torch::cat(tensors);
}

} // namespace

at::Tensor RiemannInterpolationImpl::forward(const SparseTensor& input,
                                              const at::Tensor& field) {
  return forward_full(input, field).output;
}

InterpolationResult
RiemannInterpolationImpl::forward_full(const SparseTensor& input,
                                       const at::Tensor& field) {
  TORCH_CHECK(field.dim() == 2 &&
                  field.size(1) == input.dimension() + 1,
              "Interpolation field must have shape [N, dimension + 1]");
  TORCH_CHECK(field.scalar_type() == at::kFloat ||
                  field.scalar_type() == at::kDouble,
              "Interpolation coordinates must be float32 or float64");
  TORCH_CHECK(field.device() == input.C().device(),
              "Interpolation field and coordinate map must share a device");

  const auto maps = input.coordinate_manager()->interpolation_map_weight(
      field.contiguous(), input.coordinate_map_key());
  TORCH_CHECK(maps.size() == 3,
              "Coordinate manager returned an invalid interpolation map");
  auto map_options = torch::TensorOptions()
                         .dtype(torch::kInt32)
                         .device(input.F().device());
  auto weight_options =
      input.F().options().requires_grad(false);
  auto in_map = maps[0].to(map_options);
  auto out_map = maps[1].to(map_options);
  auto weights = maps[2].to(weight_options);

  auto output = torch::zeros({field.size(0), input.F().size(1)},
                             input.F().options());
  if (in_map.numel() > 0) {
    const auto input_rows = in_map.to(torch::kLong);
    const auto output_rows = out_map.to(torch::kLong);
    auto contributions =
        input.F().index_select(0, input_rows) * weights.unsqueeze(1);
    output = output.index_add(0, output_rows, contributions);
  }
  (void)return_kernel_map_;
  (void)return_weights_;
  return {std::move(output), std::move(in_map), std::move(out_map),
          std::move(weights)};
}

} // namespace riemann
