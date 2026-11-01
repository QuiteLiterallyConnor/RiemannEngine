---
title: Home
layout: home
nav_order: 1
description: GPU-first C++ sparse tensors, adaptive voxelization and 3D Gaussian splats on LibTorch and CUDA.
permalink: /
---

# RiemannEngine
{: .fs-9 }

A native C++20 library for sparse 3D deep learning on LibTorch and CUDA. It covers sparse
convolutional networks, adaptive (mixed-resolution) voxelization and differentiable 3D Gaussian
splatting, and needs no Python.
{: .fs-6 .fw-300 }

[Get started](quickstart/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Download a release](https://github.com/QuiteLiterallyConnor/RiemannEngine/releases){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[API reference](api/){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What's inside

| Area | Highlights | Guide |
|---|---|---|
| **Sparse tensors** | `SparseTensor` / `TensorField` with batched integer coordinates, coordinate manager, quantization modes, dense/sparse conversion | [Sparse tensors & layers](sparse-tensors/) |
| **Sparse networks** | Convolution, transposed and generative convolution, pooling, batch/instance norm, about 30 activations, residual and squeeze-excite blocks, broadcast and union ops | [Sparse tensors & layers](sparse-tensors/) |
| **Adaptive voxelization** | Mixed-resolution octree-style leaves (`AdaptiveSparseTensor`), refinement by region/density/variance/predicate, adaptive convolution and pooling, GPU sphere voxelizer | [Adaptive voxelization](adaptive/) |
| **3D Gaussian splats** | `GaussianTensor` / `GaussianModel`, PLY I/O, SH degree 0-3, fused CUDA tile rasterizer, a 3DGS-style trainer with densification and pruning | [Gaussian splats](gaussian/) |
| **Training** | Optimizer factory, sparse losses, datasets and collation, gradient accumulation, LR scheduling, model and tensor persistence | [Training](training/) |

## Where to start

1. **[Installation](installation/)**: install a prebuilt MSI or `.deb`, or use the Docker image.
2. **[Quick start](quickstart/)**: build and run your first program in a few minutes.
3. **[Docker](docker/)**: develop and deploy in the GPU-ready container, with nothing installed on the host.
4. **[Core concepts](concepts/)**: coordinates, tensor strides and the coordinate manager. Read this before writing networks.
5. **[Examples](examples/)**: complete classification, segmentation, adaptive and Gaussian programs.

## Supported platforms

| OS | CUDA | LibTorch | Packages |
|---|---|---|---|
| Windows 10/11 x64 | 13.2, 13.4 | 2.15.0 | `.msi`, `.zip` |
| Ubuntu 22.04 x64 | 12.4 | 2.5.1 (CXX11 ABI) | `.deb`, `.tar.gz`, Docker `cu124` |
| Ubuntu 24.04 x64 | 13.2, 13.4 | 2.15.0 | `.deb`, `.tar.gz`, Docker `cu132` / `cu134` |

See [Installation](installation/) for the details of each variant.

## A taste

```cpp
#include <riemann.hpp>
#include <iostream>

int main() {
    auto coords = riemann::batched_coordinates(
        {torch::randint(0, 32, {1000, 3}, torch::kInt32)}, torch::kInt32, torch::kCUDA);
    riemann::SparseTensor x(torch::rand({coords.size(0), 3}, torch::kCUDA), coords);

    riemann::RiemannConvolution conv(3, 32, /*kernel_size=*/3, /*stride=*/2);
    conv->to(torch::kCUDA);
    auto y = conv->forward(x);           // stride-2 sparse convolution on the GPU
    std::cout << y.F().sizes() << "\n";  // [M, 32]
}
```
