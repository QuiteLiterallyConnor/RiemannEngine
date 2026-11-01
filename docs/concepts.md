---
title: Core concepts
nav_order: 5
description: Coordinates, batch indices, tensor strides, coordinate maps and the coordinate manager.
permalink: /concepts/
---

# Core concepts
{: .no_toc }

Everything in RiemannEngine's sparse API builds on a small set of ideas. Learn these first and the
rest of the library follows.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## Sparse tensors

A dense 3D grid of 512³ voxels holds 134 million cells, yet a LiDAR scan or surface mesh occupies
well under 1% of them. A **sparse tensor** stores only the occupied cells:

- **coordinates** `C`: an integer matrix `[N, 1 + D]`, one row per occupied cell.
- **features** `F`: a floating matrix `[N, channels]`, row-aligned with `C`.

```
   C (int32)              F (float32)
 b   x   y   z          ch0   ch1   ch2
[0,  4,  1,  7]   ->   [0.2,  0.9,  0.1]
[0,  5,  1,  7]   ->   [0.4,  0.3,  0.8]
[1, 12,  0,  3]   ->   [0.7,  0.5,  0.6]
```

Sparse convolutions compute outputs only at occupied cells, so cost scales with `N` rather than with
the volume.

## Coordinates

- **Column 0 is the batch index.** Columns `1..D` are spatial. `D` is the spatial
  dimension, usually 3, but any `D ≥ 1` works.
- Coordinates are **integers** (`torch::kInt32`) on the same device as the features.
- Coordinates must be **unique** within a batch item. Duplicates are merged when the tensor is
  built, according to the [quantization mode](#quantization).

You rarely write the batch column by hand. `batched_coordinates` builds it from per-sample
coordinate lists:

```cpp
auto coords = riemann::batched_coordinates(
    {cloud_a, cloud_b},          // each [Ni, D] int tensor
    torch::kInt32, torch::kCUDA); // -> [Na + Nb, 1 + D], column 0 = 0 or 1
```

`sparse_collate` does the same for coordinates, features and labels together. See
[batching](../sparse-tensors/#batching-and-quantization).

### From continuous points to coordinates

Real data is continuous (metres, millimetres). Divide by a voxel size and floor, or let
`sparse_quantize` do it and also deduplicate:

```cpp
auto q = riemann::sparse_quantize(points_xyz, features, labels,
                                  /*ignore_label=*/-100, /*quantization_size=*/0.05);
// q.coordinates : unique voxel coords
// q.features    : one row per voxel
// q.labels      : conflicting labels in a voxel become ignore_label
// q.inverse_mapping[i] : voxel index of input point i
```

## Tensor stride

Every coordinate map has a **tensor stride**: one integer per spatial axis that says how far apart
neighbouring cells are, measured in input (stride-1) units.

- Input tensors have stride `{1, 1, 1}`.
- A stride-2 convolution or pooling produces stride `{2, 2, 2}`. Its coordinates are multiples of 2,
  so they stay in the same coordinate frame as the input.
- A stride-2 **transposed** convolution goes back from `{2,2,2}` to `{1,1,1}`.

```
stride 1:  x = 0 1 2 3 4 5 6 7      (input)
stride 2:  x = 0   2   4   6        (after conv stride=2)
stride 4:  x = 0       4            (after another stride=2)
```

Because the frame never changes, coarse and fine features line up exactly. That makes U-Net style
skip connections cheap. Read the current stride with `x.tensor_stride()`.

## The coordinate manager

`CoordinateManager` owns all coordinate maps of a network. It is a hash table from coordinates to
rows, one per `(tensor stride, id)`, plus caches of the **kernel maps** that convolutions need.

- Constructing `SparseTensor(features, coords)` without a manager creates a new one. The backend
  (CPU or CUDA) follows the coordinates' device.
- Every layer output shares its input's manager. Strided layers insert new maps into the same
  manager, so the encoder and decoder of one forward pass share one manager.
- A **`CoordinateMapKey`** names one map: its tensor stride plus an optional string id.
  `x.coordinate_map_key()` returns it, and `x.coordinate_manager()` returns the manager.

```cpp
auto manager = std::make_shared<riemann::CoordinateManager>(
    /*D=*/3, riemann::CoordinateMapBackend::CUDA);
riemann::SparseTensor a(fa, coords, manager);
riemann::SparseTensor b(fb, a.coordinate_map_key(), manager); // same map, new features
```

Use the **key-based constructor** above to attach new features to an existing map, for example
`SparseTensor(head(x.F()), x.coordinate_map_key(), x.coordinate_manager())`. Building from raw
coordinates again would re-hash them and create a new map.

{: .important }
> Elementwise operations such as `riemann::cat`, `sum`, `mean` and `+=`, and broadcast layers, need
> their inputs to share **the same manager and the same key**. Feed them tensors that come from the
> same forward pass at the same stride.

## Quantization

When several input rows map to the same cell, `SparseTensorQuantizationMode` decides how their
features merge:

| Mode | Behaviour |
|---|---|
| `RANDOM_SUBSAMPLE` *(SparseTensor default)* | Keep one row per cell, the first occurrence |
| `UNWEIGHTED_AVERAGE` *(TensorField default)* | Mean of all rows in the cell |
| `UNWEIGHTED_SUM` | Sum of all rows |
| `MAX_POOL` | Per-channel max |
| `NO_QUANTIZATION` | Features passed through unchanged (row counts must already match) |

```cpp
riemann::SparseTensor x(features, coords, nullptr,
                        riemann::SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE);
```

## Tensor fields: keeping per-point detail

`TensorField` holds **continuous** (float) coordinates, for example raw points, and quantizes them
to a sparse lattice on request. It remembers the point-to-voxel mapping, so you can bring
voxel predictions back to the original points:

```cpp
// coords_f: [N, 1 + 3] float, column 0 = batch index
riemann::TensorField field(point_features, coords_f);
riemann::SparseTensor voxels = field.sparse();         // stride 1, averaged per voxel
auto logits = network(voxels);                          // SparseTensor at stride 1
auto per_point = logits.F().index_select(
    0, field.inverse_mapping(logits.coordinate_map_key()).to(torch::kLong)); // [N, classes]
```

`field.splat()` instead distributes each point trilinearly over the 2^D surrounding lattice
cells.

## Kernel maps and regions

A sparse convolution with kernel size `k` looks at the `k^D` offsets around each output cell
(`RegionType::HYPER_CUBE`). For each offset, the **kernel map** lists the `(input row, output row)`
pairs that exist. RiemannEngine builds kernel maps on the GPU and caches them in the manager, so
layers that share a stride and kernel shape reuse them.

`ConvolutionMode` picks the GEMM strategy: `DEFAULT` (automatic), `DIRECT_GEMM` or `COPY_GEMM`.
`RiemannAlgorithm::MEMORY_EFFICIENT` and `SPEED_OPTIMIZED` trade memory for speed on the CUDA
backend.

## Devices and dtypes

- The coordinates, features and module parameters of one forward pass must live on one device.
  Move modules with `module->to(device)` and tensors with `x.to(device)`.
- Features are usually `float32`. `x.to(torch::kFloat64)`, `x.float_()` and `x.double_()` convert
  them.
- `CoordinateMapBackend::CPU` and `CUDA` select the manager's hash table implementation. CPU-only
  builds (`CPU_ONLY=ON`) support only the CPU backend.

## Autograd

Features are ordinary `at::Tensor`s, so autograd works as usual. Gradients flow through every
layer to module parameters and, if `requires_grad` is set, to the input features. Coordinates are
not differentiable. See [Training](../training/).
