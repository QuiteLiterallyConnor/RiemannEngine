---
title: Gaussian voxelization
nav_order: 9
description: Differentiable Gaussian voxelization, LiDAR targets, and camera-to-occupancy training.
permalink: /gaussian-voxel/
---

# Gaussian-to-voxel conversion
{: .no_toc }

`riemann::gaussian_voxel` connects
`riemann::gaussian::GaussianTensor` to `riemann::adaptive::AdaptiveSparseTensor`.
CPU/CUDA voxel accumulation and backward, automatic budgeted refinement,
LiDAR ray targets, masked occupancy loss, Python bindings, and a
scene-separated camera-to-occupancy pipeline are implemented.
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

## Data flow

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

Rasterization never mutates the reference topology. CUDA keeps continuous
Gaussian data, association search, and accumulation on device. Only grid
metadata is uploaded; trainable Gaussian arrays are not downloaded.
Refinement downloads selected leaf IDs to build a new host topology.

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

The implementation supports CPU/CUDA `float32`/`float64` `GaussianTensor`
inputs and a 3D adaptive reference topology. Optional confidence is
`[N]`/`[N,1]`, nonnegative, finite, and on the same device/dtype. The output preserves
the reference topology and replaces its features with Gaussian-aggregated
features; density and occupancy are separate tensors. `max_associations`
bounds both candidate support-index cells and emitted Gaussian/voxel pairs.
CUDA support requires `CPU_ONLY=OFF`. A device-side binary search over sorted
leaf coordinates at each present level finds support associations without an
all-Gaussian/all-voxel scan. Count/prefix-sum/emission passes enforce the
association bound before allocating pair arrays. Fused float32/float64 kernels
accumulate density and weighted features, with a matching first-order backward.
Requests for higher-order CUDA gradients fail explicitly.
LibTorch differentiates covariance inversion/determinants and quaternion/scale
parameterization around those kernels. Atomic reductions are not bitwise
deterministic. Only scalar validation/count synchronizations and discrete
metadata transfers cross the host/device boundary.

The topology and LiDAR target builder remain CPU-resident. Reference features
are not read by the voxelizer; output features follow the Gaussian device.
Batched scene IDs are supported by the target
builder, while the Gaussian voxelizer currently consumes one shared topology.
Topology refinement is discrete, not a differentiable part of the forward pass.

## Automatic budgeted refinement

```python
import riemannengine as re

# reference.config.max_depth must permit the requested refinement depth.
policy = re.gaussian_voxel.AdaptiveRefinementOptions()
policy.max_depth = 2
policy.max_voxels = 30000
policy.max_splits_per_iteration = 256
policy.max_iterations = 2
policy.criterion = re.gaussian_voxel.RefinementCriterion.OccupancyUncertainty
policy.score_threshold = 0.25
policy.min_density = 1e-4

result = re.gaussian_voxel.rasterize_adaptive(
    gaussians, reference, refinement=policy
)
mixed_grid = result.rasterized.voxels
print(result.leaf_counts, result.stop_reason)
```

C++ exposes the same options and `rasterize_adaptive` in
`gaussian_voxel/refine.hpp`, also exported by `riemann.hpp`.
For each pilot pass, the policy ranks leaves by either:

- `Density`: integrated density mass divided by cell volume;
- `OccupancyUncertainty`: `4*p*(1-p)` using the selected Gaussian density-to-
  occupancy mapping, **not** the learned decoder's probabilities.

Only scores strictly above `score_threshold`, density at least `min_density`,
and levels below `max_depth` qualify. Highest scores split first, with stable
canonical-row tie breaking. Splitting replaces a parent with all eight children
and costs exactly seven additional leaves. Each iteration recomputes scores.
No source points or LiDAR targets enter this policy. The final topology is
rasterized again with autograd enabled; pilot passes retain no autograd graph.

`max_voxels` must accommodate the input grid. `max_depth` cannot exceed the
reference configuration. `max_splits_per_iteration`, `max_iterations` and the
voxelizer's `max_associations` bound refinement and support work. An association
overflow is an explicit error, never a silently truncated output.
`leaf_counts` includes the input and every accepted split pass. `stop_reason`
reports the first stopping condition checked (`threshold`, `max_depth`,
`max_voxels`, or `max_iterations`); an iteration-limit result is not a
convergence claim. Input topology/features are unchanged, children do not
overlap, and domain volume is preserved.

Refinement improves integration resolution where the current Gaussian field
provides a signal; it cannot discover geometry absent from that field.
Midpoint integration can still miss a narrow Gaussian between sample centers.
Uncertainty changes with leaf volume because occupancy maps integrated mass.

## CUDA builds

Use a CUDA-enabled LibTorch SDK for native C++, or a CUDA-enabled PyTorch
installation for Python. A toolkit alone does not turn a CPU Torch wheel into a
CUDA wheel. Build with the matching toolkit/compiler:

```powershell
cmake -S . -B build-cuda -DCPU_ONLY=OFF -DCMAKE_PREFIX_PATH=D:\libtorch_cu132 -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --config Release --target gaussian_tests
$env:PATH = "D:\libtorch_cu132\lib;$env:PATH"
ctest --test-dir build-cuda -C Release -R gaussian_tests --output-on-failure
```

For Python, configure with `-DRIEMANN_BUILD_PYTHON=ON` using the intended
Python interpreter; CMake resolves LibTorch from that interpreter's Torch
package. Rebuild/install the `riemann_python` target as in the CPU instructions
below. `re.gaussian_voxel.cuda_available()` reports whether the extension
contains CUDA kernels; also check `torch.cuda.is_available()` for runtime
availability. CPU-only builds still work without CUDA headers or libraries.

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

## Camera-to-occupancy training pipeline

The reusable model, loss, metrics, split and checkpoint helpers live in
`python/riemannengine/camera_occupancy.py`. The command-line entry point is
`examples/gaussian_voxel_nuscenes.py`. This replaces the original one-step
smoke example, not the native fixed-topology voxelization API.

The model has this explicit forward interface:

```python
logits, voxelized, gaussians = model(
    images, camera_from_reference, intrinsics, reference_grid
)
```

Only calibrated RGB images enter the model. A shared image encoder projects
multi-view features into a 3D grid and predicts one Gaussian per initial voxel.
Means can move by one full voxel size per axis, and rotations, scales,
opacities and learned features are trainable. Native differentiable Gaussian
voxelization feeds a spatial 3D convolutional decoder. Fourier position
encoding supplies location context, but no per-sample trainable embeddings or
LiDAR points are model inputs.

Training uses the mean BCE for each present class, averaged across occupied
and free classes, plus an observed-only soft-Dice term. This prevents the
majority free class from dominating the objective. Unknown cells are excluded
from both occupancy losses and accuracy metrics; they are **not empty labels**.
The targets remain LiDAR observations, not semantic annotations or
ground-truth complete 3D reconstructions.

### Supervision and reconstruction-error fixes

Without constraints in unknown space, early models filled it with roof slabs
4-12 m above the road; 55% of their predicted occupied cells were unobserved.
The example now applies four measured corrections:

- **Ego-body filtering.** About a quarter of every `LIDAR_TOP` scan hits the
  ego vehicle itself. Those returns labeled the cells around the cameras as
  occupied, so camera overlays were covered by faces. Returns inside an ego-frame
  box (`EGO_BODY_BOX`) are dropped for every sweep.
- **Ground-aligned grid.** Road returns lie at ego z of about -0.5 to 0.5 m.
  With origin z = -4, the voxel boundary at z = 0 put the ground in a 2 m box
  that contained the cameras (z of about 1.5 m). The default origin is now
  (-40, -40, -5): the road occupies [-1, 1) and the cameras the layer above.
- **Neighbouring sweeps as free space.** `--lidar-sweeps N` (default 20 per
  direction, about +/-1 s) ray-casts neighbouring sweeps from their own sensor
  poses in the reference ego frame. They add only free cells; occupied cells
  come from the keyframe, which also wins conflicts. Moving objects therefore
  do not smear into occupied labels. The gain is modest: observed fraction
  rises from 31% to 36% on a moving frame and barely changes when stationary.
- **Column-ceiling prior.** `column_ceiling_free_prior` marks unknown cells
  above the highest LiDAR return in their column. Columns without a return
  use the sample's highest return. `--ceiling-prior-weight` (default 0.1)
  adds a weak free BCE on those cells only; `--ceiling-margin` leaves extra
  levels unconstrained. The prior never touches observed cells, and other
  unknown cells remain unconstrained.

`--position-encoding height_range` (default for new runs) encodes only height
and horizontal range. The legacy `absolute` x/y/z encoding let the decoder
memorize a per-cell dataset layout. Checkpoints record the encoding, and older
checkpoints load as `absolute`. Evaluation also reports `unknown_predicted_occupied`
and `above_ceiling_predicted_occupied`, so false structure in unobserved space
is counted rather than hidden by the observed mask.

### Build and train

Use the updated Python bindings with Gaussian support enabled, PyTorch, NumPy
and Pillow. For an existing configured CPU build:

```powershell
cmake --build build-cpu --config Release --target riemann_python
cmake --install build-cpu --config Release --component python --prefix python
$env:PYTHONPATH = "$PWD\python"
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode train --steps 1000
```

`--device cpu` is the default. `--device cuda` moves the network, images,
calibration, Gaussian computation and decoder to CUDA. Topology construction,
LiDAR ray labeling and PNG rendering stay on the CPU; observed masks transfer
to the model device for loss/metrics. An unavailable CUDA device/build is an
explicit error, not a fallback.
The default grid is 40 x 40 x 8 cells of size 2 m, with origin (-40, -40, -5).
`--grid-shape`, `--grid-origin`, `--voxel-size` and `--image-size WIDTH HEIGHT`
can be set for a new run. Geometry and image size are restored from checkpoints
for evaluation, inference and resume.

Training and validation are split by **scene token**, never adjacent frames.
The deterministic seed-7 split holds out two of the mini dataset's ten scenes.
This is a custom scene-disjoint split, not an official nuScenes leaderboard
benchmark. Training samples are randomly drawn; `--batch-size` accumulates gradients
over scenes. `--max-train-samples` and `--max-eval-samples` are explicit debug
limits, round-robin across scenes, not defaults.

### Resume, evaluate and infer

```powershell
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode train --resume --steps 1000
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode eval
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode infer --sample-index 120
```

The default checkpoint is `outputs/camera_occupancy/latest.pt`. A sibling
`latest_best.pt` retains the best validation occupied IoU. Versioned checkpoints
contain model and optimizer state, optimizer step, RNG states, grid and image
configuration, exact sample tokens, scene split, selected sample subsets, and
metric history. Refinement configuration and CUDA RNG state (for CUDA runs)
are also saved; older version-1 checkpoints default to no refinement.
CPU resume is tested for bitwise-identical next updates. CUDA atomic reductions
may differ slightly even with restored RNG state.
`--steps` means additional steps when resuming. Existing
checkpoints are never silently overwritten by a fresh training run; use
`--resume` or a different `--checkpoint`. Old smoke-example state dictionaries
are intentionally rejected.

Evaluation reports observed-cell occupied IoU, free IoU, mean IoU, precision,
recall, ordinary BCE and confusion counts. Dataset metrics aggregate cell
counts across all evaluated frames, rather than averaging per-frame IoUs.
It also reports the always-occupied baseline. Evaluation rejects training-scene
sample selection; ordinary inference may use any sample. Inference reads
camera and pose/calibration metadata, **never LiDAR return files or targets**.

### Adaptive camera integration

```powershell
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode train --device cuda --adaptive-refinement --refine-depth 2 --refine-max-voxels 30000 --refine-splits 256 --refine-iterations 2 --refine-threshold 0.25
```

The policy defaults to uncertainty; `--refine-criterion density` instead uses
volume-normalized density. `--refine-min-density` gates nearly empty leaves.
Use a voxel budget at least as large as the initial grid (12,800 by default).
Resume restores the saved policy automatically; explicitly supplied refinement
settings must match it. Evaluation/inference can explicitly enable refinement
on an older fixed-grid checkpoint, but this changes integration and is **not**
guaranteed to improve that model without retraining.

The camera model predicts Gaussians from the original anchors, rasterizes them
on the refined topology, and sums child density mass and density-weighted
features back onto the base grid. Its existing spatial Conv3D decoder, labels,
metrics and PNGs therefore retain their original uniform-grid shape. This is
adaptive **Gaussian integration**, not a new mixed-resolution occupancy decoder.
An additional `*_adaptive.npz` exports actual mixed-resolution centers,
coordinates, levels, per-leaf sizes, features, density and `gaussian_occupancy`;
that last field is not the decoder prediction. The JSON reports leaf counts and
the stopping condition. Topology choices do not receive gradients.

### Learning acceptance, not just file generation

```powershell
python examples\gaussian_voxel_nuscenes.py --root D:\nuScenes --mode overfit --steps 200 --eval-every 50 --grid-shape 20 20 8 --grid-origin -20 -20 -5 --image-size 192 108 --checkpoint outputs\overfit\latest.pt --output outputs\overfit\prediction.npz
```

This deliberately fits one training-scene frame. It must reach occupied IoU
of at least 0.8 and reduce ordinary BCE by at least 40% from initialization.
Failure writes the report and exits unsuccessfully; a rendered PNG is not a
learning acceptance criterion. `--minimum-overfit-iou` can tighten the IoU
requirement. This diagnostic is **not evidence of generalization**.

On the local nuScenes mini data, a 200-step CPU overfit run on the 20 x 20 x 8
grid reached observed occupied IoU 1.000, reducing BCE from 0.702 to 0.000078
over 1,246 observed cells. A separate 201-step run using 32 frames across eight
training scenes reached occupied IoU 0.523, free IoU 0.638 and mean IoU 0.580
over **all 81 frames in the two held-out scenes**. The always-occupied baseline
was 0.401 occupied IoU. These are short, coarse-grid baseline runs, not a
converged production reconstruction model. They predate the supervision fixes
above and used the legacy grid origin and targets.

The fixes were measured with a same-seed A/B on the 20 x 20 x 8 grid, origin
z = -5 and ego-filtered targets. Every row covers all 81 held-out frames. Rows
1-3 train 600 steps on 64 frames; row 4 trains 1,500 steps on all 323 training
frames. "Unknown pred." counts predicted occupied unobserved cells, and
"above ceiling" counts the subset above the column ceiling (75,174 such cells).

| Model | Occ. IoU, keyframe | Occ. IoU, +/-20 sweeps | Free IoU, sweeps | Unknown pred. | Above ceiling |
|---|---|---|---|---|---|
| Control: no sweeps/prior, absolute encoding | 0.499 | 0.476 | 0.670 | 38,594 | 11,591 |
| Sweeps + ceiling prior, absolute encoding | 0.489 | 0.476 | 0.685 | 10,017 | 292 |
| + height/range encoding | 0.495 | 0.481 | 0.682 | 10,701 | 505 |
| + all training frames, 1,500 steps | 0.510 | 0.499 | 0.674 | 9,109 | 243 |

The prior removes 97-98% of above-ceiling structure and about three quarters
of all unobserved predictions. Observed accuracy stays within run-to-run
noise. The roof slabs and camera-enclosing faces are gone in the renders.

**Known remaining error.** The model still relies weakly on the images. Zeroing
all six images changes the mean probability by only 0.07-0.09. Training labels
have the left curb side (2-6 m, camera height) occupied 37% of the time,
versus 15% on the right, and Singapore traffic drives on the left. The models
therefore still predict a near-left wall even where LiDAR marks it free (mean
probability 0.44-0.56). Larger image encoders, higher input resolution, more
data or augmentation are the next steps; the supervision fixes do not solve this.

### Rendered outputs and limitations

`--output` writes probabilities, Gaussian density, voxel centers/coordinates,
voxel size and threshold to an `.npz`. A sibling `.json` records metrics,
split, history and image paths; `--report` changes its location. PNGs go into
the output path without its extension, or `--visualization-dir`:

- `*_voxel_geometry.png`: solid occupied voxel faces colored by height. With
  targets, separate panels show all predictions, predictions restricted to the
  **true observed mask for diagnosis only**, and LiDAR occupied cells.
- `*_camera_voxels.png`: calibrated camera images overlaid with depth-ordered
  solid predicted voxel faces. These are occupancy projections, not synthesized
  RGB images or photorealistic novel views.
- `*_occupancy_bev.png`: predicted max-over-height occupancy, log-density,
  LiDAR targets when available, and an overlay, with a legend.
- `*_occupancy_slices.png`: log-density slices with a common nonzero p99
  display scale. This is diagnostic density, not decoder occupancy.
- `*_camera_gaussians.png`: projected Gaussian centers, explicitly a debugging
  overlay rather than Gaussian splat rendering.

Unobserved space remains only weakly constrained by the ceiling prior. The
all-predictions panels deliberately show its false structures too; neither
evaluation masks nor auto-scaled colors are used to claim they are accurate.
LiDAR targets and a coarse 2 m grid cannot establish complete geometry or
small-object fidelity.

```powershell
python -m pytest python\tests\test_camera_occupancy.py python\tests\test_gaussian_voxel_visualization.py
```

Tests verify scene isolation, exact metric counts and unknown masking, gradients
through images/Gaussian geometry/native voxelization/decoder, identical next
updates after checkpoint resume, and fitting two conflicting synthetic scenes
that require different image-conditioned outputs (each IoU >= 0.95). They also
cover the ceiling prior mask and loss, rotation invariance of the height/range
encoding with legacy checkpoint loading, ego-body filtering, and sweep transforms
contributing free space only. Renderer
tests inspect actual geometry pixels, and a loader test forbids LiDAR reads
during inference. None of these tests require a local nuScenes installation.

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

The harness also accepts `--device cuda`, `--backward` and
`--adaptive-refinement --max-voxels 100000`. CUDA timings synchronize after
warm-up and each measured iteration, and include support lookup/metadata
transfer overhead, not just asynchronous launch time.

The CUDA suite is exercised on an RTX 3080 with CUDA 13.2 and LibTorch
2.15.0+cu132: 58 native Gaussian tests pass without skips. NVIDIA Compute
Sanitizer 13.4 memcheck reports zero errors for the three CUDA voxel tests.
The installed 13.2 checker instead reports internal cuBLAS library kernel-load
errors, so its run is not counted as clean validation.
Tests compare CPU/CUDA forward values and gradients for both
floating dtypes, both feature reductions and both occupancy mappings,
including rotated anisotropic Gaussians, zero confidence, nondefault streams,
empty support, and automatic topology selection. CPU tests additionally
check refinement budgets, depth/threshold/iteration stopping, volume
preservation, unchanged input topology, reproducibility and final-pass
gradients. An analytic Gaussian integral test checks that two split passes
reduce midpoint mass error by over 100x (absolute error below 0.005), and that
finer support still enforces the association budget.
The camera test covers adaptive integration, mass conservation,
image/geometry gradients and checkpoint restoration.

Local Python uses CPU-only Torch: its CUDA camera test is skipped, rather than
mistakenly reported as GPU-tested (40 Python tests pass, one skips).
All four native CPU CTest suites pass. Real nuScenes CPU adaptive training,
resume and camera-only inference have been exercised; a 3,200-leaf camera
inference grew to 3,424 then 3,648 leaves with a 32-split/two-pass policy.
The accuracy numbers above are the earlier **fixed-grid** runs, not new
adaptive/CUDA benchmark claims.

Still outstanding:

1. A learned mixed-resolution occupancy decoder and supervision on its leaves;
   the camera example still decodes on its base grid.
2. CUDA LiDAR ray traversal and topology caching/entirely device-side topology
   updates; current host topology uploads and scalar checks have overhead.
3. Higher-order CUDA autograd, deterministic reductions, and larger-scale
   accuracy/performance characterization.
