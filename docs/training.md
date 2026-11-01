---
title: Training
nav_order: 9
description: Optimizers, sparse losses, datasets, training loops, gradient accumulation, schedulers and checkpoints.
permalink: /training/
---

# Training
{: .no_toc }

Sparse networks are ordinary LibTorch modules, so any LibTorch training loop works.
`riemann::training` adds helpers for the common pieces.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

```cpp
#include <riemann.hpp>   // or "engine/learning/training.hpp"
using namespace riemann;
namespace tr = riemann::training;
```

## A plain LibTorch loop

```cpp
struct NetImpl : torch::nn::Module {
    RiemannConvolution c1{nullptr}, c2{nullptr};
    RiemannBatchNorm bn{nullptr};
    RiemannReLU relu;
    RiemannGlobalAvgPooling pool;
    RiemannLinear head{nullptr};
    NetImpl(int in, int classes) {
        c1   = register_module("c1", RiemannConvolution(in, 32, 3, 1));
        bn   = register_module("bn", RiemannBatchNorm(32));
        c2   = register_module("c2", RiemannConvolution(32, 64, 2, 2));
        head = register_module("head", RiemannLinear(64, classes));
        register_module("relu", relu);
        register_module("pool", pool);
    }
    SparseTensor forward(const SparseTensor& x) {
        auto y = relu->forward(bn->forward(c1->forward(x)));
        y = relu->forward(c2->forward(y));
        return head->forward(pool->forward(y));       // one row per batch item
    }
};
TORCH_MODULE(Net);

Net net(3, 10);
net->to(torch::kCUDA);
torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(1e-3));

for (int epoch = 0; epoch < 20; ++epoch) {
    net->train();
    for (auto& [coords, feats, labels] : batches) {          // coords [N, 4] int32, on CUDA
        SparseTensor x(feats, coords);
        auto logits = net->forward(x);
        auto loss = tr::cross_entropy_sparse(logits, labels); // labels [B] int64
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
}
```

{: .important }
> Global pooling outputs **one row per batch index in ascending order**. Labels for a
> classification head are `[B]`. For per-voxel tasks, labels must line up with the **output rows**.
> Deduplicate coordinates first with `sparse_quantize` or `sparse_collate` so the row order is
> stable.

## Optimizers

```cpp
tr::OptimizerConfig cfg;          // defaults: AdamW, lr 1e-3, weight_decay 1e-4
cfg.type = tr::OptimizerType::SGD;
cfg.lr = 0.1; cfg.momentum = 0.9; cfg.nesterov = true;
std::unique_ptr<torch::optim::Optimizer> opt = tr::create_optimizer(net->parameters(), cfg);
```

| Field | Default | Used by |
|---|---|---|
| `type` | `AdamW` | `SGD`, `Adam`, `AdamW` |
| `lr` | `1e-3` | all |
| `weight_decay` | `1e-4` | all |
| `momentum`, `nesterov` | `0.9`, `false` | SGD |
| `beta1`, `beta2`, `eps` | `0.9`, `0.999`, `1e-8` | Adam, AdamW |

## Losses

| Function | Input | Target |
|---|---|---|
| `tr::cross_entropy_sparse(logits, targets, weight = nullopt)` | `SparseTensor` `[N, K]` | `[N]` int64 |
| `tr::bce_with_logits_sparse(logits, targets, pos_weight = 1.0)` | `SparseTensor` `[N, 1]` or `[N, K]` | same shape, float |
| `tr::sparse_loss(output, target, criterion)` | `SparseTensor` | anything `criterion(output.F(), target)` accepts |
| `riemann::functional::*_loss` | `SparseTensor` | see [Functional API](../sparse-tensors/#functional-api) |

```cpp
torch::nn::SmoothL1Loss crit;
auto loss = tr::sparse_loss(pred, target_features, crit);
```

## Datasets and batches

```cpp
tr::SparseDataset ds;
ds.add_sample({coords_i /*[Ni, 3] int*/, feats_i /*[Ni, C]*/, labels_i});

std::vector<tr::SparseSample> chunk = {ds.get(0), ds.get(1)};
tr::SparseBatch b = tr::sparse_collate_samples(chunk);
// b.coordinates [ΣNi, 4] with batch index prepended, b.features, b.labels, b.batch_size
```

`SparseDataset` is a `torch::data::datasets::Dataset`, so it works with LibTorch data loaders that
use a custom collate. For raw tensors, see `riemann::sparse_collate` and `SparseCollation` in
[Batching](../sparse-tensors/#batching-and-quantization).

## Built-in loops

The loop helpers drive a `torch::nn::AnyModule` whose `forward` takes and returns a `SparseTensor`.

```cpp
torch::nn::AnyModule model(net);
auto criterion = [](const at::Tensor& out, const at::Tensor& y) {
    return torch::nn::functional::cross_entropy(out, y);
};

// One batch, your manager
auto mgr = std::make_shared<CoordinateManager>(3, CoordinateMapBackend::CUDA);
tr::StepResult s = tr::train_step(model, batch, criterion, *opt, mgr);   // {loss, num_points}

// Many batches (new manager per batch)
tr::EpochResult e = tr::train_epoch(model, batches, criterion, *opt);    // {avg_loss, total_points, num_batches}
tr::EpochResult v = tr::evaluate(model, val_batches, criterion);         // no_grad
```

{: .warning }
> `train_epoch` creates a **CPU** coordinate manager for every batch, so its batches must be on the
> CPU. For GPU training, call `train_step` with a CUDA manager or write the loop yourself.
> `evaluate` picks the backend from the batch device.

## Gradient accumulation

```cpp
tr::GradientAccumulator acc(*opt, /*accumulation_steps=*/4);
for (auto& b : batches) {
    auto loss = compute_loss(b);
    acc.backward_and_maybe_step(loss);   // scales by 1/4; steps and zeros every 4th call
}
acc.flush();                             // apply any leftover partial accumulation
```

## Learning-rate schedule

```cpp
tr::StepLRScheduler sched(*opt, /*step_size=*/30, /*gamma=*/0.1);
for (int epoch = 0; epoch < 90; ++epoch) {
    train_one_epoch();
    sched.step();                        // lr *= gamma every 30 epochs
    std::cout << sched.current_lr() << "\n";
}
```

Any LibTorch scheduler works as well.

## Checkpoints

```cpp
riemann::save_model(net, "ckpt.pt");
torch::save(*opt, "opt.pt");                 // optimizer state (LibTorch)

Net restored(3, 10);
riemann::load_model(restored, "ckpt.pt");
torch::load(*opt, "opt.pt");
```

## Mixed precision and devices

- The features (`F()`) define dtype and device. Coordinates are always integer and live on the
  same device.
- float32 is the tested path. float64 also works, for example for gradient checks. Half precision
  is not dispatched by the kernels.
- Call `net->eval()` for inference so `RiemannBatchNorm` uses its running statistics, and wrap the
  calls in `torch::NoGradGuard`.

## Autograd internals

Every sparse layer is backed by a `torch::autograd::Function` (`ConvolutionFunction`,
`LocalPoolingFunction`, `GlobalPoolingFunction`, `InterpolationFunction`, ... in
`autodiff/compat.hpp`). Advanced users can call these directly, along with the sparse matrix
helpers `SPMMFunction::apply(rows, cols, vals, dim_i, dim_j, mat)` and `SPMMAverageFunction`
(`autodiff/sparse_matmul_func.hpp`).
