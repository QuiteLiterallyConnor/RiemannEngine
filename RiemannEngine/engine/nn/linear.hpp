#ifndef RIEMANN_ENGINE_NN_LINEAR_HPP
#define RIEMANN_ENGINE_NN_LINEAR_HPP

#include "engine/tensor/sparse_tensor.hpp"

#include <torch/torch.h>

namespace riemann {

class RiemannLinearImpl : public torch::nn::Cloneable<RiemannLinearImpl> {
public:
  RiemannLinearImpl(int64_t in_features, int64_t out_features,
                    bool bias = true)
      : in_features_(in_features), out_features_(out_features),
        with_bias_(bias) {
    linear_ = register_module(
        "linear", torch::nn::Linear(torch::nn::LinearOptions(
                                    in_features_, out_features_)
                                        .bias(with_bias_)));
    reset();
  }

  void reset() override {
    if (linear_) linear_->reset_parameters();
  }

  SparseTensor forward(const SparseTensor& input) {
    TORCH_CHECK(input.F().size(1) == in_features_,
                "Input feature count does not match linear layer");
    auto output = linear_->forward(input.F());
    return SparseTensor(std::move(output), input.coordinate_map_key(),
                        input.coordinate_manager());
  }

  torch::nn::Linear linear_{nullptr};

private:
  int64_t in_features_;
  int64_t out_features_;
  bool with_bias_;
};
TORCH_MODULE(RiemannLinear);

using RiemannDenseImpl = RiemannLinearImpl;
TORCH_MODULE(RiemannDense);

} // namespace riemann

#endif
