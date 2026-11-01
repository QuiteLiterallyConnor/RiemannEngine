# RiemannEngine Python bindings

`riemannengine` exposes the RiemannEngine C++ core to Python: sparse tensors
and sparse convolutional networks, adaptive mixed-resolution voxelization,
and 3D Gaussian splats. All tensors are regular `torch.Tensor`s, gradients
flow through PyTorch autograd, and layer parameters work with `torch.optim`
optimizers.

## Installation

The extension compiles against the **torch already installed in your
environment** (CPU or a specific CUDA variant), so build without isolation:

```bash
pip install torch                      # or a CUDA variant of torch
pip install scikit-build-core ninja
pip install --no-build-isolation .    # from the repository root
```

Requirements: Python ≥ 3.9, CMake ≥ 3.24, a C++20 compiler (MSVC 2022 or
GCC 11+). If your torch has no CUDA, the build automatically falls back to
the CPU-only engine; with a CUDA torch, the matching CUDA toolkit must be
installed to compile the engine's kernels (pass
`-C cmake.define.CPU_ONLY=ON` to skip them).

## Quick start

### Sparse tensors and networks

```python
import torch
import riemannengine as re

coords = torch.randint(0, 10, (128, 3), dtype=torch.int32)
coords = torch.cat([torch.zeros(128, 1, dtype=torch.int32), coords], dim=1)
feats = torch.randn(128, 4)

x = re.SparseTensor(feats, coords)

conv = re.nn.RiemannConvolution(4, 8, kernel_size=3, dimension=3)
relu = re.nn.RiemannReLU()
pool = re.nn.RiemannGlobalAvgPooling()

y = pool(relu(conv(x)))        # SparseTensor [batch, 8]
y.F.sum().backward()           # gradients flow into conv.parameters()

optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
optimizer.step()
```

### Adaptive voxelization

```python
points = torch.rand(10_000, 3, dtype=torch.float64) * 4
features = torch.randn(10_000, 2)

config = re.adaptive.AdaptiveVoxelConfig(
    base_voxel_size=[1.0, 1.0, 1.0], max_depth=3)
t = re.adaptive.AdaptiveSparseTensor.from_points(points, features, config)
t.refine_by_density(64)        # refine crowded leaves only

weight = torch.randn(27, 2, 8, dtype=torch.float64)
out = re.adaptive.convolution(
    t, weight, re.adaptive.AdaptiveKernelConfig(kernel_size=3))
values = re.adaptive.interpolate(t, points[:16])
```

### Gaussian splats

```python
import math

g = re.gaussian.load_ply("scene.ply")
camera = re.gaussian.Camera.look_at(
    eye=[0, 0, -5], target=[0, 0, 0], up=[0, -1, 0],
    width=512, height=512, fov_x_radians=math.radians(60))
out = re.gaussian.render(g, camera)
image = out.image              # [H, W, 3] float32

model = re.gaussian.GaussianModel.from_points(positions, colors)
trainer = re.gaussian.GaussianTrainer(model)
result = trainer.step(camera, target_image)
```

## Package layout

| Module | Contents |
|---|---|
| `riemannengine` | `SparseTensor`, `TensorField`, `CoordinateManager`, enums, batching helpers (`batched_coordinates`, `sparse_collate`), functional ops (`cat`, `sum`, `mean`, `var`, `to_sparse`, …) |
| `riemannengine.nn` | `RiemannConvolution`, `RiemannConvolutionTranspose`, `RiemannLinear`, `RiemannBatchNorm`, poolings, 25+ activations, `BasicBlock`/`Bottleneck`, squeeze-excitation blocks, broadcast/pruning/union ops |
| `riemannengine.adaptive` | `AdaptiveSparseTensor`, `AdaptiveVoxelConfig`, `AxisAlignedRegion`, `convolution`, `pool`, `interpolate`, refinement/coarsening |
| `riemannengine.gaussian` | `GaussianTensor`, `GaussianModel`, `Camera`, `render`, PLY I/O, `GaussianTrainer`, `l1_loss`, `ssim` |

## Development

Build the extension directly with CMake (faster iteration than pip):

```bash
cmake -S . -B build-python -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DRIEMANN_BUILD_PYTHON=ON -DRIEMANN_BUILD_TESTS=OFF \
    -DRIEMANN_BUILD_ADAPTIVE_TESTS=OFF \
    -DPython_EXECUTABLE=$(which python)
cmake --build build-python
cp build-python/python/_C.* python/riemannengine/
PYTHONPATH=python python -m pytest python/tests -v
```
