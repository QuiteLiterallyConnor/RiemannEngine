#ifndef RIEMANN_ENGINE_NN_RESAMPLE_HPP
#define RIEMANN_ENGINE_NN_RESAMPLE_HPP

#include "engine/tensor/sparse_tensor.hpp"

#include <torch/torch.h>

namespace riemann {

struct InterpolationResult {
  at::Tensor output;
  at::Tensor in_map;
  at::Tensor out_map;
  at::Tensor weights;
};

class RiemannInterpolationImpl : public torch::nn::Module {
public:
  explicit RiemannInterpolationImpl(bool return_kernel_map = false,
                                    bool return_weights = false)
      : return_kernel_map_(return_kernel_map),
        return_weights_(return_weights) {}
  at::Tensor forward(const SparseTensor& input, const at::Tensor& field);
  InterpolationResult forward_full(const SparseTensor& input,
                                   const at::Tensor& field);

private:
  bool return_kernel_map_;
  bool return_weights_;
};
TORCH_MODULE(RiemannInterpolation);

} // namespace riemann

#endif
