#ifndef RIEMANN_ENGINE_NN_POOLING_HPP
#define RIEMANN_ENGINE_NN_POOLING_HPP

#include "engine/tensor/sparse_tensor.hpp"
#include "engine/grid/stencil.hpp"

#include <torch/torch.h>

namespace riemann {

class RiemannPoolingBaseImpl : public torch::nn::Module {
public:
  RiemannPoolingBaseImpl(int kernel_size, int stride = 1, int dilation = 1,
                         PoolingMode::Type mode = PoolingMode::LOCAL_AVG_POOLING,
                         int dimension = 3);
  RiemannPoolingBaseImpl(KernelGenerator kernel_generator,
                         PoolingMode::Type mode, int dimension);
  SparseTensor forward(const SparseTensor& input);

private:
  KernelGenerator kernel_generator_;
  PoolingMode::Type mode_;
  int dimension_;
};

class RiemannAvgPoolingImpl : public torch::nn::Module {
public:
  RiemannAvgPoolingImpl(int kernel_size, int stride = 1, int dilation = 1,
                        int dimension = 3)
      : pooling_(kernel_size, stride, dilation,
                 PoolingMode::LOCAL_AVG_POOLING, dimension) {}
  SparseTensor forward(const SparseTensor& input) {
    return pooling_.forward(input);
  }
private:
  RiemannPoolingBaseImpl pooling_;
};
TORCH_MODULE(RiemannAvgPooling);

class RiemannSumPoolingImpl : public torch::nn::Module {
public:
  RiemannSumPoolingImpl(int kernel_size, int stride = 1, int dilation = 1,
                        int dimension = 3)
      : pooling_(kernel_size, stride, dilation,
                 PoolingMode::LOCAL_SUM_POOLING, dimension) {}
  SparseTensor forward(const SparseTensor& input) {
    return pooling_.forward(input);
  }
private:
  RiemannPoolingBaseImpl pooling_;
};
TORCH_MODULE(RiemannSumPooling);

class RiemannMaxPoolingImpl : public torch::nn::Module {
public:
  RiemannMaxPoolingImpl(int kernel_size, int stride = 1, int dilation = 1,
                        int dimension = 3)
      : pooling_(kernel_size, stride, dilation,
                 PoolingMode::LOCAL_MAX_POOLING, dimension) {}
  SparseTensor forward(const SparseTensor& input) {
    return pooling_.forward(input);
  }
private:
  RiemannPoolingBaseImpl pooling_;
};
TORCH_MODULE(RiemannMaxPooling);

class RiemannPoolingTransposeImpl : public torch::nn::Module {
public:
  RiemannPoolingTransposeImpl(int kernel_size, int stride, int dilation = 1,
                              bool expand_coordinates = false,
                              int dimension = 3);
  RiemannPoolingTransposeImpl(
      KernelGenerator kernel_generator, int dimension,
      PoolingMode::Type mode = PoolingMode::LOCAL_SUM_POOLING);
  SparseTensor forward(const SparseTensor& input);

private:
  KernelGenerator kernel_generator_;
  int dimension_;
  PoolingMode::Type mode_ = PoolingMode::LOCAL_SUM_POOLING;
};
TORCH_MODULE(RiemannPoolingTranspose);

class RiemannGlobalPoolingImpl : public torch::nn::Module {
public:
  explicit RiemannGlobalPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX)
      : mode_(mode) {}

  SparseTensor forward(const SparseTensor& input);

private:
  PoolingMode::Type mode_;
};
TORCH_MODULE(RiemannGlobalPooling);

class RiemannGlobalSumPoolingImpl : public torch::nn::Module {
public:
  explicit RiemannGlobalSumPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX)
      : pooling_(RiemannGlobalPooling(mode)) {}
  SparseTensor forward(const SparseTensor& input) {
    return pooling_->forward(input);
  }

private:
  RiemannGlobalPooling pooling_{nullptr};
};
TORCH_MODULE(RiemannGlobalSumPooling);

class RiemannGlobalAvgPoolingImpl : public torch::nn::Module {
public:
  explicit RiemannGlobalAvgPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX)
      : pooling_(RiemannGlobalPooling(mode)) {}
  SparseTensor forward(const SparseTensor& input) {
    return pooling_->forward(input);
  }

private:
  RiemannGlobalPooling pooling_{nullptr};
};
TORCH_MODULE(RiemannGlobalAvgPooling);

class RiemannGlobalMaxPoolingImpl : public torch::nn::Module {
public:
  explicit RiemannGlobalMaxPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX)
      : pooling_(RiemannGlobalPooling(mode)) {}
  SparseTensor forward(const SparseTensor& input) {
    return pooling_->forward(input);
  }

private:
  RiemannGlobalPooling pooling_{nullptr};
};
TORCH_MODULE(RiemannGlobalMaxPooling);

} // namespace riemann

#endif
