---
title: Sparse tensors & layers
nav_order: 6
description: SparseTensor, TensorField, every network layer, functional ops, batching and persistence.
permalink: /sparse-tensors/
---

# Sparse tensors & layers
{: .no_toc }

A practical guide to the uniform sparse API: tensors, layers, functional operations, batching and
I/O. Exact signatures are in the [API reference](../api/).
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## Headers and namespaces

```cpp
#include <riemann.hpp>    // everything (Gaussian APIs only if RIEMANN_ENABLE_GAUSSIAN)
```

| Namespace | Contents |
|---|---|
| `riemann` | Tensors, coordinate manager, layers, batching, persistence |
| `riemann::functional` | Stateless feature ops (`relu`, `softmax`, `linear`, losses, ...) |
| `riemann::training` | Optimizers, sparse losses, datasets, loops. See [Training](../training/) |
| `riemann::adaptive` | Adaptive voxelization. See [Adaptive](../adaptive/) |
| `riemann::gaussian` | Gaussian splats. See [Gaussian splats](../gaussian/) |

The canonical headers for individual pieces are `engine/tensor/*.hpp`, `engine/nn/*.hpp`,
`engine/helpers/*.hpp` and `persistence.hpp`. The `layers/*.hpp` and `helpers/*.hpp` headers
forward to them.

## SparseTensor

```cpp
SparseTensor(at::Tensor features, at::Tensor coordinates,
             std::shared_ptr<CoordinateManager> manager = nullptr,
             SparseTensorQuantizationMode::Type quantization_mode =
                 SparseTensorQuantizationMode::RANDOM_SUBSAMPLE,
             GPUMemoryAllocatorBackend::Type alloc_type = GPUMemoryAllocatorBackend::PYTORCH,
             RiemannAlgorithm::Mode algo = RiemannAlgorithm::DEFAULT);

SparseTensor(at::Tensor features, CoordinateMapKey key,
             std::shared_ptr<CoordinateManager> manager);   // reuse an existing map
```

- `features`: `[N, C]` floating point. `coordinates`: `[N, 1 + D]` integer, batch first, on the
  same device.
- Duplicate coordinates are merged using `quantization_mode`, so `F().size(0)` can be smaller than
  `N`. See [Quantization](../concepts/#quantization).

### Accessors

| Method | Returns |
|---|---|
| `F()` | Features `[N, C]` (a reference, so autograd tracks it) |
| `C()` | Coordinates `[N, 1 + D]`, computed lazily from the manager and then cached |
| `tensor_stride()` | `std::vector<int>`-like stride per axis |
| `dimension()` | Spatial dimension `D` |
| `coordinate_map_key()`, `coordinate_manager()` | Map identity and owner |
| `device()`, `dtype()`, `requires_grad()` | Feature properties |
| `decomposed_coordinates()`, `decomposed_features()`, `decomposed_coordinates_and_features()` | Per-batch-item lists |

### Conversions

```cpp
auto y = x.to(torch::kCUDA);          // coordinates + features + new CUDA manager
auto d = x.to(torch::kFloat64);       // feature dtype only; also float_(), double_()
auto n = x.detach();                  // detached features, same map

at::Tensor dense = x.dense();         // [B, C, X, Y, Z] starting at the minimum coordinate
at::Tensor fixed = x.dense(std::vector<int64_t>{32, 32, 32},
                           torch::zeros({3}, torch::kInt32)); // fixed extent from origin 0
at::Tensor coo = x.sparse();          // torch sparse COO tensor, shape [B, X, Y, Z, C]
```

Going the other way:

```cpp
riemann::SparseTensor s  = riemann::to_sparse(dense_bcxyz);       // drop all-zero cells
riemann::SparseTensor s2 = riemann::to_sparse_all(dense_bcxyz);   // keep every cell
```

### Arithmetic

```cpp
auto z = a + b;     // also -, *, / : union of both coordinate sets, missing rows = 0
auto w = a * 0.5;   // scalar ops keep the map
a += b;             // in place: same manager AND same key required
```

`a + b` (and `-`, `*`, `/`) aligns the two tensors on the **union** of their coordinates.
The result lives in a **new** coordinate manager. To combine tensors without leaving the
current manager, for example in skip connections, use `riemann::cat`, `sum` and `mean` on tensors
that share a key, or `+=`.

## TensorField

```cpp
TensorField(at::Tensor features, at::Tensor coordinates /* float [N, 1+D] */,
            std::shared_ptr<CoordinateManager> manager = nullptr,
            SparseTensorQuantizationMode::Type mode = UNWEIGHTED_AVERAGE);

SparseTensor sparse(stride_type tensor_stride = {}, ..., mode = UNWEIGHTED_AVERAGE) const;
SparseTensor splat() const;                       // trilinear splat to lattice corners
at::Tensor inverse_mapping(const CoordinateMapKey& sparse_key) const; // point -> voxel row
```

Use a `TensorField` when you need per-point outputs, as in semantic segmentation of raw points. See
[Tensor fields](../concepts/#tensor-fields-keeping-per-point-detail).

## Layers

All layers are LibTorch module holders (`TORCH_MODULE`). Construct them by value, call
`->forward(x)`, and move them with `->to(device)`. Parameters show up in `->parameters()`. Unless
noted, `forward` takes and returns a `SparseTensor`.

### Convolutions

```cpp
RiemannConvolution(int in, int out, int kernel_size = 3, int stride = 1, int dilation = 1,
                   bool bias = false, bool expand_coordinates = false,
                   ConvolutionMode::Type mode = ConvolutionMode::DEFAULT, int dimension = 3);
RiemannConvolutionTranspose(/* same parameters */);
RiemannGenerativeConvolutionTranspose(int in, int out, int kernel_size = 3, int stride = 1,
                   int dilation = 1, bool bias = false,
                   ConvolutionMode::Type mode = ConvolutionMode::DEFAULT, int dimension = 3);
```

| Layer | Output coordinates |
|---|---|
| `RiemannConvolution`, stride 1 | Same as the input (same key) |
| `RiemannConvolution`, stride s | Input coordinates rounded down to multiples of `stride × s`; stride becomes `stride × s` |
| `RiemannConvolutionTranspose`, stride s | Back to stride `stride / s`, reusing the existing finer map. Use it in decoders |
| `RiemannGenerativeConvolutionTranspose` | Generates **new** coordinates: every kernel offset of every input cell. Use it to grow shapes |
| `expand_coordinates = true` | Output also includes kernel-reachable cells that were empty |

The weights are `base()->kernel_` with shape `[K^D, in, out]`, and `base()->bias_`.

```cpp
riemann::RiemannConvolution down(32, 64, /*k=*/2, /*stride=*/2);
riemann::RiemannConvolutionTranspose up(64, 32, /*k=*/2, /*stride=*/2);
auto coarse = down->forward(x);  // tensor_stride {2,2,2}
auto fine   = up->forward(coarse); // tensor_stride {1,1,1}, same key as x
```

### Pooling

```cpp
RiemannMaxPooling(int kernel_size, int stride = 1, int dilation = 1, int dimension = 3);
RiemannAvgPooling(...);   RiemannSumPooling(...);
RiemannPoolingTranspose(int kernel_size, int stride, int dilation = 1,
                        bool expand_coordinates = false, int dimension = 3);  // unpooling

RiemannGlobalAvgPooling();   RiemannGlobalMaxPooling();   RiemannGlobalSumPooling();
RiemannGlobalPooling(PoolingMode::Type mode = PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX);
```

Global pooling returns **one row per batch item** (batch coordinate, origin), which is what a
classifier head needs.

### Normalization

| Layer | Notes |
|---|---|
| `RiemannBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true)` | Over all rows; respects `train()` / `eval()` |
| `RiemannSyncBatchNorm` | Alias of `RiemannBatchNorm` |
| `RiemannInstanceNorm(num_features)` | Per batch item |
| `RiemannStableInstanceNorm(num_features)` | Numerically safer variance |
| `RiemannLayerNorm`, `RiemannGroupNorm` | Wrap the LibTorch modules, applied to features |

### Activations and feature modules

Each wraps the LibTorch module of the same name and takes the same constructor options:

`RiemannReLU`, `RiemannReLU6`, `RiemannLeakyReLU`, `RiemannELU`, `RiemannSELU`, `RiemannCELU`,
`RiemannGELU`, `RiemannSiLU`, `RiemannSigmoid`, `RiemannTanh`, `RiemannHardsigmoid`,
`RiemannHardswish`, `RiemannHardshrink`, `RiemannHardtanh`, `RiemannLogSigmoid`, `RiemannPReLU`,
`RiemannRReLU`, `RiemannSoftplus`, `RiemannSoftshrink`, `RiemannSoftsign`, `RiemannTanhshrink`,
`RiemannThreshold`, `RiemannSoftmin`, `RiemannSoftmax`, `RiemannLogSoftmax`, `RiemannDropout`,
`RiemannAlphaDropout`, plus `RiemannSinusoidal(in, out)`.

```cpp
riemann::RiemannLeakyReLU act(torch::nn::LeakyReLUOptions().negative_slope(0.2));
```

### Linear

```cpp
RiemannLinear(int64_t in_features, int64_t out_features, bool bias = true); // alias RiemannDense
```

Applied row by row. Coordinates are preserved.

### Residual and squeeze-excite blocks

```cpp
BasicBlock(int inplanes, int planes, int stride = 1, int dilation = 1,
           torch::nn::AnyModule downsample = {}, double bn_momentum = 0.1, int dimension = 3);
Bottleneck(/* same */);            // expansion = 4
SELayer(int channel, int reduction = 16, int dimension = 3);
SEBasicBlock(int inplanes, int planes, int stride = 1, int dilation = 1,
             torch::nn::AnyModule downsample = {}, int reduction = 16,
             double bn_momentum = 0.1, int dimension = 3);
SEBottleneck(/* same */);
```

When `stride != 1` or `inplanes != planes * expansion`, pass a `downsample` that maps the input to
the block's output shape and stride:

```cpp
torch::nn::AnyModule down(riemann::RiemannConvolution(32, 64, /*k=*/1, /*stride=*/2));
riemann::BasicBlock block(32, 64, /*stride=*/2, /*dilation=*/1, down);
```

### Structural operations

| Layer / function | Purpose |
|---|---|
| `riemann::cat({a, b, ...})` | Channel concatenation (same key) |
| `riemann::sum`, `mean`, `var` | Elementwise reduction across tensors (same key) |
| `RiemannUnion()->forward({a, b})` | Sum over the **union** of coordinate sets within one manager |
| `RiemannPruning()->forward(x, mask)` | Keep only rows where `mask` is true |
| `RiemannBroadcastAddition`, `RiemannBroadcastMultiplication`, `RiemannBroadcastConcatenation` | Combine per-row features with a global (per-batch) tensor, for example from global pooling |
| `RiemannInterpolation()->forward(x, points)` | Trilinear interpolation of features at continuous `points` `[M, 1 + D]` |
| `RiemannToDenseTensor(shape)`, `RiemannToSparseTensor()`, `RiemannToFeature()` | Conversions inside a `Sequential` |

```cpp
// Squeeze-excite style gating with broadcast:
riemann::RiemannGlobalAvgPooling gap;
riemann::RiemannLinear fc(64, 64);
riemann::RiemannBroadcastMultiplication gate;
auto g = riemann::functional::sigmoid(fc->forward(gap->forward(x)));
auto y = gate->forward(x, g);
```

## Functional API

`riemann::functional` applies an op to `F()` and rewraps the result with the same map:

```cpp
using namespace riemann::functional;
auto y = relu(x);
auto p = softmax(x, /*dim=*/1);
auto z = dropout(x, 0.1, /*training=*/true);
auto w = linear(x, weight, bias);
auto v = wrap(x, my_tensor_op(x.F()));   // rewrap any feature tensor
```

Available ops: `relu`, `leaky_relu`, `elu`, `selu`, `celu`, `gelu`, `sigmoid`, `tanh`, `hardtanh`,
`hardswish`, `relu6`, `hardsigmoid`, `silu`, `logsigmoid`, `softplus`, `softsign`, `softshrink`,
`hardshrink`, `tanhshrink`, `softmax`, `softmin`, `log_softmax`, `glu`, `prelu`, `rrelu`,
`threshold`, `linear`, `dropout`, `alpha_dropout`, `normalize`, `batch_norm`.

The loss helpers return `at::Tensor`: `cross_entropy`, `nll_loss`, `binary_cross_entropy`,
`binary_cross_entropy_with_logits`, `mse_loss`, `l1_loss`, `smooth_l1_loss`, `kl_div`,
`hinge_embedding_loss`.

## Batching and quantization

```cpp
// Coordinates only
at::Tensor batched_coordinates(const std::vector<at::Tensor>& coords,
                               c10::ScalarType dtype = torch::kInt32,
                               c10::optional<torch::Device> device = {});

// Coordinates + features (+ labels)
auto [coords, feats, labels] = riemann::sparse_collate(coord_list, feat_list, label_list);

// Continuous points -> unique voxels (+ inverse map)
riemann::SparseQuantizeResult q = riemann::sparse_quantize(
    points, features, labels, /*ignore_label=*/-100, /*quantization_size=*/0.02);
// q.coordinates [M, D] int, q.features / q.labels (optional), q.unique_index [M],
// q.inverse_mapping [N]: voxel row for every input point

// Unique rows of an integer coordinate matrix
auto [unique_rows, inverse] = riemann::unique_coordinate_map(coords);
```

`SparseCollation(limit_numpoints, dtype, device)` is a reusable collate functor that drops samples
once the batch exceeds `limit_numpoints`.

## Persistence

{% raw %}
```cpp
riemann::save_model(net, "net.pt");            // any torch::nn module or holder
riemann::load_model(net, "net.pt");            // construct with the same shapes first

riemann::save_sparse_tensor(x, "x.pt");
riemann::SparseTensor x2 = riemann::load_sparse_tensor("x.pt");   // fresh manager
riemann::save_sparse_tensors({a, b}, "ab.pt");

riemann::save_tensors({{"weights", w}, {"mask", m}}, "misc.pt");
auto named = riemann::load_tensors("misc.pt");
```
{% endraw %}

`load_sparse_tensor_data` and `reconstruct_sparse_tensor(data, manager)` let you load into an
existing manager.

## Enumerations

| Enum | Values |
|---|---|
| `SparseTensorQuantizationMode` | `RANDOM_SUBSAMPLE`, `UNWEIGHTED_AVERAGE`, `UNWEIGHTED_SUM`, `MAX_POOL`, `NO_QUANTIZATION` |
| `ConvolutionMode` | `DEFAULT`, `DIRECT_GEMM`, `COPY_GEMM` |
| `PoolingMode` | `LOCAL_{SUM,AVG,MAX}_POOLING`, `GLOBAL_{SUM,AVG,MAX}_POOLING_{DEFAULT,KERNEL,PYTORCH_INDEX}` |
| `BroadcastMode` | `ELEMENTWISE_ADDITON`, `ELEMENTWISE_MULTIPLICATION` |
| `RegionType` | `HYPER_CUBE`, `HYPER_CROSS`, `CUSTOM` |
| `CoordinateMapBackend` | `CPU`, `CUDA` |
| `RiemannAlgorithm` | `DEFAULT`, `MEMORY_EFFICIENT`, `SPEED_OPTIMIZED` |
| `GPUMemoryAllocatorBackend` | `PYTORCH`, `CUDA` |
