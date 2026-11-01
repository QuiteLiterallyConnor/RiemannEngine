---
title: Adaptive voxelization
nav_order: 7
description: Mixed-resolution sparse tensors with octree-style refinement, adaptive convolution and the GPU sphere voxelizer.
permalink: /adaptive/
---

# Adaptive voxelization
{: .no_toc }

`riemann::adaptive` stores a scene as **non-overlapping leaves at different refinement levels**.
Detail goes only where it is needed: surfaces, dense regions or high-variance features.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## Headers

```cpp
#include <octree/adaptive.hpp>     // canonical
// #include <dynamic/dynamic.hpp>  // forwarding alias; also pulled in by <riemann.hpp>
using namespace riemann::adaptive;
```

## The model

- **Level 0** cells measure `base_voxel_size` per axis. Each level **halves** the cell extent, so a
  level-`L` cell is `base / 2^L`.
- `max_depth` caps refinement. Dimensions 1 to 4 are supported, with depth up to 30.
- A leaf is identified by an `AdaptiveVoxelKey<D>{coordinates, level}`. Its integer `coordinates`
  are in level-`L` cell units, relative to `origin`.
- Cells are half-open boxes `[lo, hi)`. The active leaves never overlap.
- **Refining** a leaf replaces it with `2^D` children. `RefinePolicy::Replicate` copies its
  features into the children, and `Split` divides them by `2^D` so sums are conserved.
- **Coarsening** merges descendants back into one leaf with `Reduction::Mean`, `Sum` or `Max`.

```
level 0 ┌───────────────┬───────────────┐
        │               │   ┌───┬───┐   │
        │      L0       │   │L2 │L2 │L1 │   <- refined only near detail
        │               │   ├───┼───┤   │
        │               │   │L2 │L2 │   │
        └───────────────┴───┴───┴───┴───┘
```

## Configuration

```cpp
struct AdaptiveVoxelConfig {
    std::vector<double> base_voxel_size;   // required, one entry per axis
    std::vector<double> min_voxel_size;    // optional; if set, must equal base / 2^max_depth
    std::vector<double> origin;            // optional; empty = zero vector
    uint32_t max_depth = 0;
    void validate() const;                 // throws std::invalid_argument
};
```

```cpp
AdaptiveVoxelConfig cfg;
cfg.base_voxel_size = {8.0, 8.0, 8.0};
cfg.min_voxel_size  = {1.0, 1.0, 1.0};
cfg.max_depth       = 3;      // 8 -> 4 -> 2 -> 1
```

## Building a tensor

```cpp
// From points: quantize to level-0 cells, averaging features per cell
auto t = AdaptiveSparseTensor::from_points(points /*[N,D] float64*/,
                                           features /*[N,C]*/, cfg);

// From explicit leaves
auto u = AdaptiveSparseTensor::from_voxels(coords /*[M,D] int64*/, levels /*[M]*/,
                                           feats /*[M,C]*/, cfg);
```

`from_points` keeps the source points. `refine_by_density`, `refine_by_variance` and
`revoxelize_from_points()` re-bin them as the topology changes.

## Refinement and coarsening

{% raw %}
```cpp
// Region-based: refine everything inside a box to level 2
t.refine(AxisAlignedRegion{{0, 0, 0}, {8, 8, 8}}, /*target_level=*/2);
t.refine(AxisAlignedRegion::everything(3), 1, RefinePolicy::Split);

// Data-driven (needs source points)
t.refine_by_density(/*points_per_leaf_threshold=*/16);
t.refine_by_variance(/*feature_variance_threshold=*/0.05);

// Arbitrary predicate on each leaf's bounds and level
t.refine_if([](int64_t leaf, const std::array<double, 4>& lo,
               const std::array<double, 4>& hi, uint32_t level) {
    return lo[2] < 1.0 && level < 3;          // refine near the floor
});

// Merge back
t.coarsen(AxisAlignedRegion::everything(3), /*target_level=*/0, Reduction::Max);
```
{% endraw %}

## Inspecting a tensor

| Method | Returns |
|---|---|
| `size()`, `dimension()`, `channels()` | Leaf count, `D`, feature width |
| `coordinates()`, `levels()` | `[N, D]` int64 cell coordinates, `[N]` levels |
| `centers()`, `bounds()`, `voxel_sizes()`, `volumes()` | World-space geometry per leaf |
| `features()`, `set_features(f)`, `with_features(f)` | Feature access (`with_features` returns a copy) |
| `counts_by_level()` | Leaves per level |
| `index_of({x,y,z}, level)`, `index_of_key(coords, level)` | Row of a leaf, or -1 |
| `has_overlapping_leaves()` | Sanity check, should be `false` |
| `topology_version()` | Increments on every topology change. Used by caches |

## Operations

### Adaptive convolution

```cpp
AdaptiveKernelConfig kc;
kc.kernel_size = 3;                           // positive, odd
kc.mode = AdaptiveConvMode::VolumeWeighted;   // CenterSampled | VolumeWeighted | OverlapWeighted
kc.output_level_shift = 0;                    // 0 = same leaves; -n = coarser; +n = subdivide

auto weight = torch::randn({27, t.channels(), 16}, torch::dtype(torch::kFloat64).requires_grad(true));
AdaptiveSparseTensor y = adaptive_convolution(t, weight, kc);   // weight [K^D, Cin, Cout]

// Transposed: maps back onto a reference topology (the forward conv's input)
AdaptiveSparseTensor back = adaptive_convolution_transpose(y, weight_t, kc, /*reference=*/t);
```

Each kernel tap sits at `output centre + offset × output cell size`.

| Mode | How neighbours of different sizes contribute |
|---|---|
| `CenterSampled` *(default)* | The leaf that contains the tap point, weight 1. On a uniform grid this matches an ordinary sparse convolution |
| `VolumeWeighted` | Same neighbour, weighted `V_in / V_out` |
| `OverlapWeighted` | Every leaf overlapping the tap box, weighted `overlap / V_out`. Most accurate, and cost grows with the level gap |

`output_level_shift = -n` lifts each output to its ancestor `n` levels up (never below level 0).
`+n` subdivides every leaf `n` times, which multiplies the output count by `2^(D·n)` and must fit
`max_depth`.

Kernel maps are cached in `KernelMapCache::instance()`, keyed by (topology id, kernel size, mode,
shift). Topology changes mint new ids, so stale maps are never served. The default capacity is
64 entries. Inspect the cache with `.stats()`, and manage it with `.set_capacity(n)` and `.clear()`.
`build_adaptive_kernel_map(t, kc)` exposes the raw `(in, out, kernel, weight)` lists.

### Pooling, resampling and interpolation

```cpp
at::Tensor g = adaptive_global_avg_pool(t, /*volume_weighted=*/true);   // [C]
auto c  = adaptive_pool(t, /*target_level=*/1, Reduction::Max);           // coarsen finer leaves
auto u  = t.resample_uniform(/*level=*/2);                                // all leaves at one level
auto r  = a.resample_to(b);                                               // onto b's topology
auto s  = r + b;                                                          // same topology -> elementwise
auto cat = concatenate(r, b);                                             // channel concat

at::Tensor v = interpolate(t, query_points /*[M,D]*/, InterpMode::Multilinear); // [M, C]; default Nearest
```

`adaptive_pool` coarsens every leaf finer than the target level. `resample_to` reduces finer
sources, replicates coarser ones and fills uncovered reference leaves with zeros. `interpolate`
returns zeros for points outside every leaf.

{: .warning }
> `resample_uniform` at a fine level can produce a very large number of leaves. Compute
> `pow(2, D*level) * size()` first to bound it.

### Sphere voxelization

```cpp
double achieved = 0.0;
auto sphere = voxelize_sphere(/*radius=*/1.0, /*target_accuracy_percent=*/99.5,
                              /*center=*/{0, 0, 0}, /*max_voxels=*/10'000'000,
                              /*max_depth=*/10, &achieved);
double vol = sphere_volume_estimate(sphere);       // compare with 4/3 π r³

auto gpu_sphere = voxelize_sphere_cuda(1.0, 99.5); // CUDA builds only
```

The voxelizer starts from one cube of side `2r` and splits the largest boundary cells first, until
`accuracy = 100 × (1 − |estimate − exact| / exact)` reaches the target. The target must be in
(0, 100). A leaf counts as occupied when its centre is inside the sphere. The result has one
float channel: 1 for occupied leaves and 0 otherwise. It throws `std::runtime_error` when the
target cannot be reached within `max_voxels` or `max_depth`.

## Devices

- `t.to(torch::kCUDA)` moves **features** to the GPU. The topology and kernel maps stay on the
  host and are built with `std::thread` workers.
- `adaptive_convolution` runs its gather/GEMM/scatter on the features' device.
- Convolution, pooling and interpolation are differentiable with respect to features and weights.

## Interop with uniform sparse tensors

There is no direct conversion function yet. To feed a uniform `SparseTensor` network, resample to
one level and wrap the coordinates:

```cpp
auto u = t.resample_uniform(2);
auto coords = torch::cat({torch::zeros({u.size(), 1}, torch::kInt64), u.coordinates()}, 1)
                  .to(torch::kInt32);
riemann::SparseTensor x(u.features().to(torch::kFloat32), coords);
```

## Limitations

| Area | Status |
|---|---|
| Hierarchy and kernel maps | Host-side (multi-threaded), not GPU |
| Uniform fast path | Not implemented. Uniform inputs still use the general adaptive path |
| `output_level_shift > 0` | Output count grows by `2^(D·n)`. A `std::length_error` is thrown if it would overflow |
| `OverlapWeighted` cost | Grows with the level gap between neighbours |
