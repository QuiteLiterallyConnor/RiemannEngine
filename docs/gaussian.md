---
title: Gaussian splats
nav_order: 8
description: GaussianTensor, trainable 3D/4D models, differentiable CPU/CUDA rasterization, training, D-NeRF datasets and PLY I/O.
permalink: /gaussian/
---

# Gaussian splats
{: .no_toc }

`riemann::gaussian` covers 3D and dynamic 4D Gaussian splatting: containers, cameras, a differentiable tile
rasterizer (LibTorch reference path plus fused CUDA kernels), training loops, D-NeRF datasets and PLY I/O.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

{: .note }
> Gaussian support is compiled when `RIEMANN_ENABLE_GAUSSIAN=ON`, which is the default and how
> every release is built. `<riemann.hpp>` then includes all of it.

## Headers

| Header | Contents |
|---|---|
| `gaussian_voxel/voxelize.hpp` | `riemann::gaussian_voxel::rasterize`, density/occupancy outputs and aggregation options |
| `gaussian_voxel/occupancy.hpp` | LiDAR ray targets, occupied/free/unknown masks, and masked occupancy loss |
| `splat/soup.hpp` | `GaussianTensor`, an activated structure-of-arrays container |
| `splat/model.hpp` | `GaussianModel`, raw trainable 3DGS parameters |
| `splat/model4d.hpp` | `Gaussian4DModel`, time-conditioned Gaussian parameters |
| `splat/trainer4d.hpp` | `Gaussian4DTrainer`, `Gaussian4DTrainerConfig` |
| `splat/dnerf.hpp` | `load_dnerf_views`, `TimedView`, `DNerfLoadOptions` |
| `splat/camera.hpp` | `Camera` |
| `splat/render.hpp` | `render`, `render_gaussians`, `RenderOptions`, `RenderOutput` |
| `splat/trainer.hpp` | `GaussianTrainer`, `GaussianTrainerConfig`, `l1_loss`, `ssim` |
| `splat/optimizer.hpp` | `GaussianOptimizer`, a lightweight Adam helper for a `GaussianTensor` |
| `splat/ply.hpp` | `load_gaussian_ply`, `load_gaussian_model_ply`, `save_gaussian_ply` |
| `splat/harmonics.hpp` | `sh_basis`, `eval_sh_color`, `sh_coefficient_count`, `kShC0` |
| `splat/primitive.hpp` | `Gaussian3D<T>`, a single analytic Gaussian (covariance, density, sampling, bounds) |

The `normal/normal_*.hpp` headers forward to these.

The separate [`riemann::gaussian_voxel` API](./gaussian-voxel.md) rasterizes
Gaussian features and density onto an existing adaptive voxel topology and
provides LiDAR occupancy supervision. The reference voxelizer and target
builder are CPU-only; CUDA voxel indexing and kernels are planned separately.

## Two representations

| | `GaussianTensor` | `GaussianModel` |
|---|---|---|
| Values | **Activated**: opacity in [0, 1], positive scales, unit quaternions, RGB colours | **Raw** 3DGS parameters: logit opacity, log scale, unnormalized quaternion, SH coefficients |
| Colour | One RGB per Gaussian | Spherical harmonics up to degree 3 (view-dependent) |
| Extras | Optional `features [N, F]` and `velocities [N, 3]` | none |
| Use it for | Editing, filtering, transforming, inference | Training with `GaussianTrainer` |
| Convert | `GaussianModel::from_tensor(t)` | `m.to_gaussian_tensor()` |

### GaussianTensor

```cpp
using namespace riemann::gaussian;
const int64_t N = 1000;
auto f = torch::kFloat32;
GaussianTensor g(
    torch::randn({N, 3}, f),                                   // positions
    torch::tensor({1.f, 0.f, 0.f, 0.f}).repeat({N, 1}),        // rotations (w, x, y, z)
    torch::full({N, 3}, 0.05f),                                // scales
    torch::full({N, 1}, 0.8f),                                 // opacities
    torch::rand({N, 3}, f));                                   // colors

auto visible = g.filter_opacity(0.1);
auto box     = g.filter_bounds(torch::full({3}, -1.f), torch::full({3}, 1.f));
auto moved   = g.transform(torch::tensor({1.f, 0.f, 0.f, 0.f}),   // rotation quaternion
                           torch::tensor({0.f, 0.f, 2.f}));        // translation
auto both    = GaussianTensor::concatenate({g, moved});
at::Tensor cov = g.covariances();                                // [N, 3, 3]
```

Every field is a contiguous tensor with one device and dtype (float32 or float64). The tensor also
has `select`, `remove`, `append`, `clone`, `to(device)`, `enable_gradients()` and
`project_constraints()`. The last one clamps scales to at least 1e-6 and opacities to [0, 1]
after a manual optimizer step.

### GaussianModel

```cpp
// From a coloured point cloud: kNN-based scales, identity rotations, constant opacity
GaussianModel m = GaussianModel::from_points(points /*[N,3]*/, rgb /*[N,3]*/,
                                             /*sh_degree=*/3, /*initial_opacity=*/0.1);
```

| Raw parameter | Shape | Activation |
|---|---|---|
| `xyz` | `[N, 3]` | none |
| `f_dc` | `[N, 3]` | colour = `0.5 + kShC0 · f_dc` |
| `f_rest` | `[N, K − 1, 3]`, `K = (deg + 1)²` | SH evaluation in `render` |
| `opacity_raw` | `[N, 1]` | sigmoid |
| `scale_raw` | `[N, 3]` | exp |
| `rotation_raw` | `[N, 4]` | normalize (w, x, y, z) |

The activated getters `positions()`, `rotations()`, `scales()`, `opacities()`, `colors()` and
`sh_coefficients()` are differentiable. `parameters()` returns the six raw tensors in the order
above.

## Cameras

```cpp
struct Camera {
    int64_t width, height;
    double fx, fy, cx, cy;
    at::Tensor world_to_camera;   // [4, 4] float32
    double near_plane = 0.01;
    static Camera from_fov(int64_t w, int64_t h, double fov_x_radians, at::Tensor world_to_camera);
    static Camera look_at(std::array<double,3> eye, std::array<double,3> target,
                          std::array<double,3> up, int64_t w, int64_t h, double fov_x_radians);
};
```

The camera uses the **OpenCV/COLMAP convention**: +x right, +y **down**, +z forward. Pixel `(i, j)`
has its centre at `(i + 0.5, j + 0.5)`. With `look_at`, pass `up = {0, -1, 0}` for a y-up world.

```cpp
auto cam = Camera::look_at({0, 0, -3}, {0, 0, 0}, {0, -1, 0}, 640, 480, /*fov_x=*/1.0);
```

## Rendering

```cpp
RenderOptions opts;
opts.background = torch::ones({3});        // white; undefined = black
RenderOutput out = render(m, cam, opts);   // or render(gaussian_tensor, cam, opts)

at::Tensor image = out.image;   // [H, W, 3]
at::Tensor depth = out.depth;   // [H, W] alpha-weighted camera z
at::Tensor alpha = out.alpha;   // [H, W] accumulated opacity
```

| `RenderOptions` field | Default | Meaning |
|---|---|---|
| `background` | undefined (black) | `[3]` colour |
| `sh_degree` | `-1` | Highest SH degree to evaluate (`-1` uses the model's degree) |
| `tile_size` | `16` | Tile side for the CPU path. The CUDA kernels always use 16 |
| `chunk_size` | `1024` | Gaussians composited per chunk inside a tile |
| `use_cuda_kernels` | `true` | Use the fused CUDA rasterizer for CUDA inputs |

`RenderOutput` also returns `means2d [N, 2]` (its `.grad()` is filled by backward), `visible [N]`
and `radii [N]`.

The renderer projects Gaussians with the EWA approximation, sorts them by depth and composites
front to back. **Gradients flow to every input.** For the low-level entry point
`render_gaussians(positions, rotations, scales, opacities, colors, cam, opts)`, all tensors must
be **float32 on the same device**.

{: .tip }
> Move the model with `m.to(torch::kCUDA)` and the CUDA kernels are picked automatically. Set
> `opts.use_cuda_kernels = false` to compare against the LibTorch reference implementation.

## Training (3DGS recipe)

```cpp
GaussianModel model = load_gaussian_model_ply("init.ply").to(torch::kCUDA);

GaussianTrainerConfig cfg;
cfg.scene_extent = 5.0;        // roughly the camera-rig radius
cfg.max_gaussians = 2'000'000;
GaussianTrainer trainer(model, cfg);   // model must outlive the trainer

for (int it = 0; it < 30000; ++it) {
    const auto& [cam, target] = views[it % views.size()];   // target: [H, W, 3] float32
    auto r = trainer.step(cam, target);
    if (it % 1000 == 0)
        std::cout << it << " loss " << r.loss << " gaussians " << r.gaussians << "\n";
}
save_gaussian_ply(model, "trained.ply");
```

Each `step` renders with the active SH degree, computes `(1 − λ)·L1 + λ·(1 − SSIM)`, runs Adam
with per-parameter learning rates and handles the schedule below.

| Config | Default | |
|---|---|---|
| `lr_position` → `lr_position_final` over `lr_decay_steps` | 1.6e-4 → 1.6e-6 over 30000 | Exponential decay, scaled by `scene_extent` |
| `lr_dc`, `lr_rest`, `lr_opacity`, `lr_scale`, `lr_rotation` | 2.5e-3, 1.25e-4, 0.05, 5e-3, 1e-3 | |
| `beta1`, `beta2`, `epsilon` | 0.9, 0.999, 1e-15 | Adam |
| `lambda_ssim` | 0.2 | |
| `sh_increase_interval` | 1000 | Active SH degree +1 every N steps |
| `densify`, `densify_start`, `densify_interval`, `densify_stop` | true, 500, 100, 15000 | |
| `densify_grad_threshold` | 2e-4 | Screen-space gradient that triggers clone or split |
| `percent_dense` | 0.01 | Above `percent_dense × extent` Gaussians split, below they clone |
| `min_opacity`, `max_scale_fraction` | 0.005, 0.1 | Pruning thresholds |
| `opacity_reset_interval` | 3000 | Clamps opacities to ≤ 0.01 |
| `max_gaussians` | 0 (unlimited) | |
| `render` | `RenderOptions{}` | Used for every training render |

For custom loops, call `accumulate_statistics(out, cam)`, `densify_and_prune()` (returns
`{added, removed}`) and `reset_opacity()` yourself. `render_image(cam)` renders without gradients.
The free functions `l1_loss(pred, target)` and `ssim(pred, target, window = 11, sigma = 1.5)`
work on `[H, W, 3]` images.

## 4D Gaussian splats (dynamic scenes)

`Gaussian4DModel` and `Gaussian4DTrainer` extend the 3D model with linear velocity and Gaussian temporal
opacity, `load_dnerf_views` reads D-NeRF/Blender datasets, `reconstruct4d` runs full-data reconstruction,
`save_four_dgs` exports `.4dgs` playback files and `examples/view_fourdgs.py` plays them back. See the
dedicated [4D Gaussian splats](./4dgs.md) page.

## Lightweight optimization of a GaussianTensor

`GaussianOptimizer` runs Adam on the fields of an activated `GaussianTensor`. You supply the
prediction (for example from `render(g, cam).image`):

```cpp
GaussianOptimizer opt(g, /*lr=*/1e-2);
for (int i = 0; i < 100; ++i) {
    opt.zero_grad();
    auto pred = render(g, cam).image;
    auto loss = GaussianOptimizer::objective(g, pred, target,
                                             /*opacity_weight=*/1e-3, /*scale_weight=*/1e-3);
    opt.step(loss);
}
```

## PLY files

```cpp
GaussianTensor t = load_gaussian_ply("scene.ply");        // activated, float32, CPU, SH rest dropped
GaussianModel  m = load_gaussian_model_ply("scene.ply");  // raw params incl. SH degree 0–3
save_gaussian_ply(m, "out.ply");                           // standard 3DGS layout
save_gaussian_ply(t, "out_dc.ply");                        // SH degree 0
```

Supported layouts are binary little-endian **standard 3DGS** and **PlayCanvas / SuperSplat
compressed** (chunk + packed + SH elements). The files are compatible with common splat viewers.
Missing or malformed files throw `std::runtime_error`.

## Spherical harmonics

```cpp
int64_t k = sh_coefficient_count(3);                    // 16
at::Tensor basis = sh_basis(3, dirs /*[N,3] unit*/);    // [N, 16]
at::Tensor rgb = eval_sh_color(3, m.sh_coefficients(), dirs);   // clamp_min(0.5 + Σ, 0)
```
