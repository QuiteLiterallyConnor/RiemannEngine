---
layout: default
title: Python bindings
nav_order: 10
---

# Python bindings
{: .no_toc }

The `riemannengine` package exposes the RiemannEngine C++ core to Python:
sparse tensors and sparse convolutional networks, adaptive mixed-resolution
voxelization, and 3D Gaussian splats. All data moves as regular
`torch.Tensor`s, gradients flow through PyTorch autograd, and layer
parameters plug into `torch.optim` optimizers.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

The extension compiles against the **torch already installed in your
environment** (CPU or a specific CUDA variant), so install torch first and
build without isolation:

```bash
pip install torch                    # or a CUDA variant of torch
pip install scikit-build-core ninja
pip install --no-build-isolation .   # from the repository root
```

Requirements: Python ≥ 3.9, CMake ≥ 3.24, and a C++20 compiler
(Visual Studio 2022 on Windows, GCC 11+ on Linux).

{: .note }
With a CPU-only torch wheel the build automatically produces a CPU-only
engine. With a CUDA torch, the matching CUDA toolkit must be installed to
compile the engine's kernels; pass `-C cmake.define.CPU_ONLY=ON` to skip
them.

## Sparse tensors and networks

```python
import torch
import riemannengine as re

coords = torch.randint(0, 10, (128, 3), dtype=torch.int32)
coords = torch.cat([torch.zeros(128, 1, dtype=torch.int32), coords], dim=1)
feats = torch.randn(128, 4)

x = re.SparseTensor(feats, coords)      # x.F, x.C, x.tensor_stride, ...

conv = re.nn.RiemannConvolution(4, 8, kernel_size=3, dimension=3)
relu = re.nn.RiemannReLU()
pool = re.nn.RiemannGlobalAvgPooling()

y = pool(relu(conv(x)))                 # SparseTensor [batch, 8]
y.F.sum().backward()                    # autograd flows into the kernels

optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
optimizer.step()
```

Batching helpers mirror the C++ API: `re.batched_coordinates`,
`re.sparse_collate` and `re.SparseCollation` build batched coordinate
tensors; `re.cat`, `re.sum`, `re.mean`, `re.var`, `re.to_sparse` and
`re.to_sparse_all` operate on whole sparse tensors.

## Adaptive voxelization

```python
points = torch.rand(10_000, 3, dtype=torch.float64) * 4
features = torch.randn(10_000, 2)

config = re.adaptive.AdaptiveVoxelConfig(
    base_voxel_size=[1.0, 1.0, 1.0], max_depth=3)
t = re.adaptive.AdaptiveSparseTensor.from_points(points, features, config)

t.refine_by_density(64)                 # refine crowded leaves only
region = re.adaptive.AxisAlignedRegion(lo=[0, 0, 0], hi=[2, 2, 2])
t.refine(region, 2)

weight = torch.randn(27, 2, 8, dtype=torch.float64)
out = re.adaptive.convolution(
    t, weight, re.adaptive.AdaptiveKernelConfig(kernel_size=3))
values = re.adaptive.interpolate(t, points[:16])
```

## Gaussian splats

```python
import math

g = re.gaussian.load_ply("scene.ply")   # standard or SuperSplat PLY
camera = re.gaussian.Camera.look_at(
    eye=[0, 0, -5], target=[0, 0, 0], up=[0, -1, 0],
    width=512, height=512, fov_x_radians=math.radians(60))
out = re.gaussian.render(g, camera)
image = out.image                       # [H, W, 3] float32

# 3DGS-style training
model = re.gaussian.GaussianModel.from_points(positions, colors)
trainer = re.gaussian.GaussianTrainer(model)
result = trainer.step(camera, target_image)
```

## Package layout

| Module | Contents |
|---|---|
| `riemannengine` | `SparseTensor`, `TensorField`, `CoordinateManager`, enums, batching helpers, functional ops |
| `riemannengine.nn` | `RiemannConvolution`, `RiemannConvolutionTranspose`, `RiemannLinear`, `RiemannBatchNorm`, poolings, 25+ activations, `BasicBlock`/`Bottleneck`, squeeze-excitation blocks, broadcast/pruning/union ops |
| `riemannengine.adaptive` | `AdaptiveSparseTensor`, `AdaptiveVoxelConfig`, `AxisAlignedRegion`, `convolution`, `pool`, `interpolate`, refinement and coarsening |
| `riemannengine.gaussian` | `GaussianTensor`, `GaussianModel`, `Camera`, `render`, PLY I/O, `GaussianTrainer`, `l1_loss`, `ssim` |

## Development builds

For faster iteration than `pip install`, build the extension directly with
CMake and point `PYTHONPATH` at the package:

```bash
cmake -S . -B build-python -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DRIEMANN_BUILD_PYTHON=ON -DRIEMANN_BUILD_TESTS=OFF \
    -DRIEMANN_BUILD_ADAPTIVE_TESTS=OFF \
    -DPython_EXECUTABLE=$(which python)
cmake --build build-python
cp build-python/python/_C.* python/riemannengine/
PYTHONPATH=python python -m pytest python/tests -v
```
