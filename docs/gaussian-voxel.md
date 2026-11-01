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
The fixed-topology CPU reference forward/backward operation is implemented;
device-side indexing, LiDAR targets, and end-to-end training remain later stages.
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
CUDA inputs fail with a clear unsupported-operation error until the device
index and accumulation kernels are implemented. Batched scene IDs and
topology refinement are not part of this first implementation.

## Implementation stages

1. **Fixed-topology forward/backward (implemented):** public options and output,
   adaptive support association, differentiable density/feature reduction,
   CPU correctness and finite-difference gradient tests. Unsupported device
   and dimensionality combinations are rejected explicitly.
2. **CUDA execution:** add device-side spatial indexing, support lookup,
   accumulation, and backward; compare against the reference implementation
   and test gradients on CUDA.
3. **Adaptive refinement integration:** add configurable, capped refinement
   criteria and mixed-resolution association without overlap or double-counting.
   Keep the resulting topology fixed during each gradient computation.
4. **LiDAR supervision:** add batched ray traversal, occupied/free/unknown
   labels, coordinate transforms, cached target support, and masked loss.
5. **Integration and evaluation:** add the synthetic end-to-end training
   example, inference path, focused benchmarks, and complete API/numerics/
   platform documentation.

Each stage is independently testable. Performance and hardware-specific
benchmark values are recorded only after the corresponding implementation has
been run on the named hardware.
