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

### Dynamic 4D Gaussian splats

```python
g = re.gaussian
options = g.DNerfLoadOptions()
options.downscale = 25
options.max_frames = 6
options.stride = 25
views = g.load_dnerf_views("tests/data/bouncingballs", "train", options)
model = g.Gaussian4DModel.from_points(
    torch.rand(256, 3) * 2 - 1, torch.full((256, 3), 0.5), sh_degree=0)
config = g.Gaussian4DTrainerConfig()
config.densify = False
config.render.background = torch.ones(3)
trainer = g.Gaussian4DTrainer(model, config)
for step in range(120):
    view = views[step % len(views)]
    result = trainer.step(view.camera, view.time, view.image)
image = trainer.render_image(views[0].camera, time=0.5)
g.save_ply(model.at_time(0.5), "snapshot.ply")  # static snapshot only

# Preserve all nine tensors to retain the dynamic model (not optimizer state).
torch.save([p.detach().cpu() for p in model.parameters()], "dynamic.pt")
restored = g.Gaussian4DModel(*torch.load("dynamic.pt", weights_only=True))

# Resume optimization, including Adam, iteration, statistics and RNG state.
trainer.save_checkpoint("training.pt")
# Construct a model and trainer with the same config, then:
trainer.load_checkpoint("training.pt")

# Dynamic playback interchange, including motion, temporal opacity and SH.
export_options = g.FourDGSExportOptions()
export_options.seconds_per_time_unit = 2.0  # normalized [0,1) -> two seconds
report = g.save_four_dgs(model, "scene.4dgs", export_options)
print(report.bytes, report.clamped_rgb_components, report.max_position_error)
```

The model uses linear per-Gaussian motion and Gaussian temporal opacity windows,
not a deformation network. Time conditioning is differentiable on CPU and CUDA.
See the [4D guide](../docs/4dgs.md) for the
parameterization, learning rates, limitations and bouncingballs test results.
For all-frame, full-resolution reconstruction with automatic checkpoints,
validation/test metrics and PNG previews, build the native `reconstruct4d`
target described in the [full-data runner guide](../docs/4dgs.md#full-data-reconstruction-runner).
The [dynamic export guide](../docs/4dgs.md#dynamic-4dgs-export) describes
avala-ai's version-1 draft `.4dgs` compatibility, time remapping, quantization
and required RGB/SH clamping. Export refuses existing destinations and does not
replace the lossless training checkpoint.
For interactive playback with orbit/pan/zoom and seamless ping-pong orbit GIFs,
see the [Python viewer guide](../docs/4dgs.md#interactive-playback-and-looping-orbit-gifs).

### Gaussian-to-voxel accumulation and automatic refinement

```python
device = "cpu"  # use "cuda" with CUDA PyTorch and a CPU_ONLY=OFF extension
config = re.adaptive.AdaptiveVoxelConfig([1.0, 1.0, 1.0], max_depth=2)
reference = re.adaptive.AdaptiveSparseTensor.from_voxels(
    torch.zeros(1, 3, dtype=torch.int64),
    torch.zeros(1, dtype=torch.int32), torch.zeros(1, 1), config)
positions = torch.full((1, 3), 0.5, device=device, requires_grad=True)
gaussians = re.gaussian.GaussianTensor(
    positions,
    torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    torch.full((1, 3), 0.4, device=device),
    torch.full((1, 1), 0.8, device=device),
    torch.zeros(1, 3, device=device),
    features=torch.ones(1, 8, device=device))
policy = re.gaussian_voxel.AdaptiveRefinementOptions()
policy.max_depth = 2
policy.max_voxels = 64
result = re.gaussian_voxel.rasterize_adaptive(
    gaussians, reference, refinement=policy)
result.rasterized.density.sum().backward()
print(result.leaf_counts, result.stop_reason)
```

For fixed topology, use `re.gaussian_voxel.rasterize`. Both paths support
float32/float64 and arbitrary feature widths. Topology decisions are discrete;
the final values remain differentiable. Reference geometry stays on CPU while
Gaussian arrays, accumulation and backward run on their selected device.
`re.gaussian_voxel.cuda_available()` reports compiled kernel support, distinct
from `torch.cuda.is_available()` runtime availability.

The [Gaussian voxelization guide](../docs/gaussian-voxel.md) explains the
refinement criteria/budgets, LiDAR supervision, CUDA builds, and the nuScenes
camera-only occupancy workflow. Its optional adaptive integration still feeds
a uniform-grid decoder; the exported adaptive Gaussian occupancy is not the
decoder's prediction.

## Package layout

| Module | Contents |
|---|---|
| `riemannengine` | `SparseTensor`, `TensorField`, `CoordinateManager`, enums, batching helpers (`batched_coordinates`, `sparse_collate`), functional ops (`cat`, `sum`, `mean`, `var`, `to_sparse`, …) |
| `riemannengine.nn` | `RiemannConvolution`, `RiemannConvolutionTranspose`, `RiemannLinear`, `RiemannBatchNorm`, poolings, 25+ activations, `BasicBlock`/`Bottleneck`, squeeze-excitation blocks, broadcast/pruning/union ops |
| `riemannengine.adaptive` | `AdaptiveSparseTensor`, `AdaptiveVoxelConfig`, `AxisAlignedRegion`, `convolution`, `pool`, `interpolate`, refinement/coarsening |
| `riemannengine.gaussian` | `GaussianTensor`, `GaussianModel`, `Gaussian4DModel`, `Camera`, `render`, PLY I/O, `save_four_dgs`, `GaussianTrainer`, `Gaussian4DTrainer`, D-NeRF dataset loading, `l1_loss`, `ssim` |
| `riemannengine.gaussian_voxel` | CPU/CUDA `rasterize`, `rasterize_adaptive`, density/feature aggregation, occupancy mapping, refinement options |
| `riemannengine.occupancy` | CPU LiDAR ray targets and observed-only binary cross entropy |
| `riemannengine.camera_occupancy` | Image-conditioned occupancy network, grid/refinement specifications, loss/metrics and checkpoints |

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
