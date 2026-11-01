---
title: Gaussian voxelization
nav_order: 9
description: Architecture and staged implementation plan for differentiable Gaussian-to-adaptive-voxel conversion.
permalink: /gaussian-voxel/
---

# Gaussian-to-voxel conversion
{: .no_toc }

`riemann::gaussian_voxel` connects
`riemann::gaussian::GaussianTensor` to `riemann::adaptive::AdaptiveSparseTensor`.
The fixed-topology CPU reference, LiDAR ray targets, masked occupancy loss,
Python bindings, and a nuScenes mini example are implemented. CUDA voxel
execution and automatic topology refinement remain future work.
{: .fs-5 .fw-300 }

## Existing engine components

- `GaussianTensor` stores activated positions, unit-quaternion rotations,
  positive scales, opacities, and optional learned features. Its covariance is
  `R diag(scale²) Rᵀ`, so valid positive scales give a positive-definite
  covariance without requiring matrix repair.
- `AdaptiveSparseTensor` stores non-overlapping leaves and their feature rows.
  Cell geometry and topology are discrete and CPU-resident; feature tensors can
  be differentiable and live on CPU or CUDA.
- Gaussian image rendering has its own 2D tile compositing and CUDA backward
  kernels. That rasterizer is not a volumetric voxel accumulator and is not
  reused for this operation.
- The engine's Gaussian and adaptive modules are included by their own headers;
  the umbrella header conditionally exports Gaussian APIs.

## Proposed data flow

```text
GaussianTensor ── position / covariance / opacity / learned features
       │
       ├── discrete support and voxel association (fixed for one forward pass)
       │
       ├── evaluate Gaussian density at each associated voxel center
       │
       └── differentiable weighted reduction
                     │
                     ├── density / occupancy values
                     └── arbitrary-width voxel feature rows
                                  │
                                  └── AdaptiveSparseTensor on the reference topology
```

For Gaussian `i` and voxel center `xᵥ`, the continuous contribution is based on

`qᵢᵥ = (xᵥ - μᵢ)ᵀ Σᵢ⁻¹ (xᵥ - μᵢ)` and
`gᵢᵥ = aᵢ exp(-qᵢᵥ / 2) / ((2π)^(3/2) sqrt(det Σᵢ))`,

where `aᵢ = opacity_density_scale × opacityᵢ × confidenceᵢ` (confidence defaults
to one). The per-voxel density mass is `ρᵥ = Σᵢ gᵢᵥ × volume(Vᵥ)`, evaluated by
voxel-center midpoint quadrature. Opacity is therefore not interpreted as
physical occupancy without the explicit `opacity_density_scale` mapping. The
support radius uses the covariance diagonal to find a conservative axis-aligned
support box and limits discrete associations; it does not alter the continuous
gradient inside that fixed support. This midpoint approximation is not exactly
mass-conserving for Gaussians under-resolved by the voxel grid.

Feature aggregation supports weighted sum and normalized weighted mean, using
`gᵢᵥ × volume(Vᵥ)` as the weight. `OccupancyModel::Poisson` maps density to
`1 - exp(-ρᵥ)`; `OccupancyModel::ClampedLinear` maps it to `clamp(ρᵥ, 0, 1)`.

## Differentiability boundary

For fixed topology and fixed Gaussian-to-voxel associations, the contribution
weights and reductions must be expressed as differentiable LibTorch operations
or a custom autograd operation with a matching backward. This allows gradients
to reach means, covariance parameters, opacity/density, confidence, and learned
features. Support selection, sparse coordinate construction, and adaptive
refinement are discrete; gradients do not pass through those decisions.

The initial correctness milestone uses the existing adaptive topology as an
immutable reference and must not mutate it. CUDA execution must keep continuous
Gaussian data and feature accumulation on device; a host-side topology/index
build may consume only discrete grid metadata, not copy trainable Gaussian
values to CPU.

## Current API and limits

```cpp
#include <gaussian_voxel/voxelize.hpp>

riemann::gaussian_voxel::GaussianVoxelOptions options;
options.support_radius = 3.0;
options.opacity_density_scale = 1.0; // application-defined opacity mapping
options.feature_aggregation =
    riemann::gaussian_voxel::FeatureAggregation::NormalizedWeightedMean;
options.occupancy_model = riemann::gaussian_voxel::OccupancyModel::Poisson;

auto voxelized =
    riemann::gaussian_voxel::rasterize(gaussians, reference_grid, options);
auto sparse_features = voxelized.voxels.features();
auto density = voxelized.density;     // [V, 1]
auto occupancy = voxelized.occupancy; // [V, 1]
```

The initial implementation supports CPU `float32`/`float64` `GaussianTensor`
inputs and a 3D adaptive reference topology. Optional confidence is CPU
`[N]`/`[N,1]`, nonnegative, finite, and the same dtype. The output preserves
the reference topology and replaces its features with Gaussian-aggregated
features; density and occupancy are separate tensors. `max_associations`
bounds both candidate support-index cells and emitted Gaussian/voxel pairs.
CUDA inputs fail with a clear unsupported-operation error. The current
voxelizer uses a CPU topology and CPU LibTorch operations; the LiDAR target
builder is also CPU-only. Batched scene IDs are supported by the target
builder, while the Gaussian voxelizer currently consumes one shared topology.
Topology refinement is an explicit preprocessing decision, not a differentiable
part of the forward pass.

## LiDAR occupancy supervision

The target builder traverses each sensor-origin-to-return segment through the
reference grid. Traversed leaves are free, a return leaf is occupied, and
occupied takes precedence where rays overlap. `observed` is the union of those
masks; all remaining leaves are unknown and are omitted from the loss.

```cpp
#include <gaussian_voxel/occupancy.hpp>

auto targets = riemann::occupancy::from_lidar(
    points, origins, ray_batch_ids, batch_size, reference_grid,
    /*options=*/{}, sensor_to_world); // optional [B, 4, 4]
auto labels = targets.labels();       // [B, V], occupied=1
auto loss = riemann::occupancy::masked_binary_cross_entropy(
    predicted_occupancy, targets);
```

`points` and `origins` are `[R, 3]` float tensors. `ray_batch_ids` is CPU
int64 `[R]`; the optional transforms are finite CPU `[B, 4, 4]` matrices.
`LidarRayOptions::max_ray_distance` clips free-space traversal without
inventing a return at the clipped endpoint. `max_ray_voxels` bounds traversal work for an individual ray.
`masked_binary_cross_entropy` accepts probabilities in `[0, 1]`, averages
across observed leaves only, and returns a differentiable zero when there are
no observations.

Python exposes matching interfaces under `riemannengine.gaussian_voxel` and
`riemannengine.occupancy`. The Python extension must be built with
`RIEMANN_ENABLE_GAUSSIAN=ON`.

## nuScenes mini example

`examples/gaussian_voxel_nuscenes.py` joins `sample_data`, calibration, sensor,
and ego-pose records from the nuScenes mini metadata, reads the six camera
keyframes and `LIDAR_TOP`, predicts image-conditioned Gaussians, rasterizes
them onto a fixed 2 m grid, and trains with the masked LiDAR occupancy loss.
It also supports image-only inference and writes a compressed `.npz` containing
voxel centers, density, and occupancy probabilities.

```powershell
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode train --steps 10
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode infer
```

Install Pillow if it is not already available (`python -m pip install Pillow`).
The example is intentionally CPU-only: it rejects a CUDA-visible PyTorch
runtime because the native 3D voxelizer and LiDAR target builder do not yet
provide CUDA execution. The example is an integration/training smoke path, not
a benchmark or a claim of converged nuScenes performance. Use
`--sample-index` to select a different keyframe.

## Validation and remaining work

The CPU C++ suite covers voxel aggregation and gradients, ray labeling,
transforms, batch isolation, clipping, unknown masks, and masked-loss
gradients. Python smoke tests cover the exposed rasterizer and occupancy API.
On an AMD Ryzen 5 5600X with PyTorch 2.15.0+cpu and a Release build, the
benchmark harness measured a 1.497 ms median (1.515 ms mean) over three timed
runs for 32 Gaussians, 1,728 voxels, and eight feature channels. One warm-up
run is excluded. This is a small CPU reference-path measurement, not an
end-to-end nuScenes or CUDA performance claim.

Run the harness with:

```powershell
python benchmarks\benchmark_gaussian_voxel.py --resolution 12 --gaussians 32 --iterations 3
```

Still outstanding:

1. CUDA spatial indexing, accumulation, and backward kernels, validated against
   the CPU reference on a CUDA-enabled LibTorch installation.
2. A configurable adaptive-refinement policy with explicit topology and work
   budgets; callers can currently refine the reference grid using the existing
   adaptive tensor API before rasterization.
