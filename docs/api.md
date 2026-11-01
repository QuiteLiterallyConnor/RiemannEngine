---
title: API reference
nav_order: 15
description: Header map and links to the generated Doxygen reference for every public class.
permalink: /api/
---

# API reference
{: .no_toc }

The full, generated reference for every header lives at **[API reference (Doxygen)](reference/)**.
This page maps what you want to do to the right header and class.
{: .fs-5 .fw-300 }

[Open the Doxygen reference](reference/){: .btn .btn-primary }
[Class list](reference/annotated.html){: .btn }
[File list](reference/files.html){: .btn }

1. TOC
{:toc}

---

## Headers

Most programs need one include:

```cpp
#include <riemann.hpp>             // sparse tensors, layers, training, adaptive, Gaussians*
#include <octree/adaptive.hpp>     // adaptive voxelization only
```

\* Gaussian headers are included when the library was built with `RIEMANN_ENABLE_GAUSSIAN` (on in
all release packages). The installed CMake target defines it for you.

Canonical headers live in `engine/`, `octree/` and `splat/`. The headers in `layers/`, `helpers/`,
`dynamic/`, `normal/`, `core/` and `learning/`, and most root-level headers, only forward to them
and remain for compatibility.

| Namespace | Contents |
|---|---|
| [`riemann`](reference/namespaceriemann.html) | Sparse tensors, coordinate managers, layers, functional ops, batching |
| [`riemann::training`](reference/namespaceriemann_1_1training.html) | Optimizers, batches, training and evaluation loops |
| [`riemann::adaptive`](reference/namespaceriemann_1_1adaptive.html) | Adaptive (octree) voxelization and convolution |
| [`riemann::gaussian`](reference/namespaceriemann_1_1gaussian.html) | 3D Gaussian splats: model, rendering, training, PLY |

## Sparse tensors

| Type / function | Header | Guide |
|---|---|---|
| [`SparseTensor`](reference/classriemann_1_1SparseTensor.html) | `engine/tensor/sparse_tensor.hpp` | [Sparse tensors](../sparse-tensors/) |
| [`TensorField`](reference/classriemann_1_1TensorField.html) | `engine/tensor/field.hpp` | [TensorField](../sparse-tensors/#tensorfield) |
| [`CoordinateManager`](reference/classriemann_1_1CoordinateManager.html) | `engine/grid/space.hpp` | [Core concepts](../concepts/) |
| [`SparseQuantizeResult`](reference/structriemann_1_1SparseQuantizeResult.html), `sparse_quantize`, `batched_coordinates`, `sparse_collate` | `helpers/batching.hpp`, `helpers/discretization.hpp` | [Sparse tensors](../sparse-tensors/) |
| `cat`, `sum`, `mean`, `var`, `to_sparse`, `to_sparse_all` | `engine/nn/operations.hpp` | [Sparse tensors](../sparse-tensors/) |
| Functional activations (`riemann::functional::relu`, `gelu`, `softmax`, ...) | `functions.hpp` | [Sparse tensors](../sparse-tensors/) |
| Arithmetic operators (`+ - * /`) | `engine/tensor/sparse_tensor.hpp` | [Arithmetic](../sparse-tensors/#arithmetic) |
| `save` / `load` helpers | `persistence.hpp` | [Training](../training/) |

## Layers

All layers are LibTorch modules (`TORCH_MODULE` holders around `...Impl` classes) in `engine/nn/`.

| Family | Classes | Header |
|---|---|---|
| Convolution | [`RiemannConvolution`](reference/classriemann_1_1RiemannConvolutionImpl.html), [`RiemannConvolutionTranspose`](reference/classriemann_1_1RiemannConvolutionTransposeImpl.html), `RiemannGenerativeConvolutionTranspose` | `engine/nn/conv.hpp` |
| Normalization | [`RiemannBatchNorm`](reference/classriemann_1_1RiemannBatchNormImpl.html), `RiemannLayerNorm`, `RiemannGroupNorm`, `RiemannInstanceNorm`, `RiemannStableInstanceNorm` | `engine/nn/normalization.hpp` |
| Activation | `RiemannReLU`, `RiemannHardsigmoid`, `RiemannHardswish`, `RiemannSinusoidal` | `engine/nn/activation.hpp` |
| Linear | `RiemannLinear`, `RiemannDense` | `engine/nn/linear.hpp` |
| Pooling | `RiemannAvgPooling`, `RiemannSumPooling`, `RiemannMaxPooling`, `RiemannPoolingTranspose`, [`RiemannGlobalPooling`](reference/classriemann_1_1RiemannGlobalPoolingImpl.html), `RiemannGlobalAvgPooling`, `RiemannGlobalSumPooling`, `RiemannGlobalMaxPooling` | `engine/nn/pooling.hpp` |
| Residual | [`BasicBlock`](reference/classriemann_1_1BasicBlockImpl.html), `Bottleneck` | `engine/nn/residual.hpp` |
| Squeeze-excite | `SELayer`, [`SEBasicBlock`](reference/classriemann_1_1SEBasicBlockImpl.html), `SEBottleneck` | `engine/nn/squeeze_excite.hpp` |
| Structure | `RiemannUnion`, `RiemannPruning`, `RiemannToSparseTensor`, `RiemannToDenseTensor`, `RiemannToFeature`, `RiemannBroadcastAddition`, `RiemannBroadcastMultiplication`, `RiemannBroadcastConcatenation` | `engine/nn/operations.hpp` |
| Interpolation | `RiemannInterpolation` | `engine/nn/resample.hpp` |

Autograd functions behind the layers (for custom modules) are in `autodiff/`, for example
`autodiff/conv_func.hpp`, `autodiff/sparse_matmul_func.hpp` (`SPMMFunction`) and
`autodiff/compat.hpp`.

## Training

| Type | Header | Guide |
|---|---|---|
| [`training::OptimizerConfig`](reference/structriemann_1_1training_1_1OptimizerConfig.html), `OptimizerType`, `StepLRScheduler` | `engine/learning/training.hpp` | [Training](../training/) |
| [`training::SparseSample`](reference/structriemann_1_1training_1_1SparseSample.html), [`training::SparseBatch`](reference/structriemann_1_1training_1_1SparseBatch.html) | `engine/learning/training.hpp` | [Training](../training/) |
| `StepResult`, `EpochResult`, `GradientAccumulator` | `engine/learning/training.hpp` | [Training](../training/) |

## Adaptive voxelization

| Type | Header | Guide |
|---|---|---|
| [`adaptive::AdaptiveSparseTensor`](reference/classriemann_1_1adaptive_1_1AdaptiveSparseTensor.html), [`adaptive::SourcePoints`](reference/structriemann_1_1adaptive_1_1SourcePoints.html) | `octree/tensor.hpp` | [Adaptive voxelization](../adaptive/) |
| [`adaptive::AdaptiveVoxelConfig`](reference/structriemann_1_1adaptive_1_1AdaptiveVoxelConfig.html), `Reduction`, `RefinePolicy`, `AxisAlignedRegion` | `octree/config.hpp` | [Adaptive voxelization](../adaptive/) |
| [`adaptive::AdaptiveKernelConfig`](reference/structriemann_1_1adaptive_1_1AdaptiveKernelConfig.html), `AdaptiveConvMode`, [`KernelMapCache`](reference/classriemann_1_1adaptive_1_1KernelMapCache.html) | `octree/stencil.hpp` | [Adaptive convolution](../adaptive/) |
| `interpolate`, `InterpMode` | `octree/ops.hpp` | [Adaptive voxelization](../adaptive/) |
| `voxelize_sphere`, `SphereFrame` | `octree/sphere.hpp` | [Adaptive voxelization](../adaptive/) |

## Gaussian splats

| Type | Header | Guide |
|---|---|---|
| [`gaussian::GaussianModel`](reference/classriemann_1_1gaussian_1_1GaussianModel.html) | `splat/model.hpp` | [Gaussian splats](../gaussian/) |
| [`gaussian::GaussianTensor`](reference/classriemann_1_1gaussian_1_1GaussianTensor.html) | `splat/soup.hpp` | [Gaussian splats](../gaussian/) |
| [`gaussian::Gaussian3D`](reference/classriemann_1_1gaussian_1_1Gaussian3D.html) | `splat/primitive.hpp` | [Gaussian splats](../gaussian/) |
| [`gaussian::Camera`](reference/structriemann_1_1gaussian_1_1Camera.html) | `splat/camera.hpp` | [Gaussian splats](../gaussian/) |
| [`gaussian::RenderOptions`](reference/structriemann_1_1gaussian_1_1RenderOptions.html), [`RenderOutput`](reference/structriemann_1_1gaussian_1_1RenderOutput.html), `render` | `splat/render.hpp` | [Gaussian splats](../gaussian/) |
| [`gaussian::GaussianOptimizer`](reference/classriemann_1_1gaussian_1_1GaussianOptimizer.html) | `splat/optimizer.hpp` | [Gaussian splats](../gaussian/) |
| [`gaussian::GaussianTrainer`](reference/classriemann_1_1gaussian_1_1GaussianTrainer.html), [`GaussianTrainerConfig`](reference/structriemann_1_1gaussian_1_1GaussianTrainerConfig.html) | `splat/trainer.hpp` | [Gaussian splats](../gaussian/) |
| PLY load/save | `splat/ply.hpp` | [Gaussian splats](../gaussian/) |
| Spherical harmonics | `splat/harmonics.hpp` | [Gaussian splats](../gaussian/) |

## Source

Browse the headers on
[GitHub](https://github.com/QuiteLiterallyConnor/RiemannEngine/tree/main/RiemannEngine).
The reference is regenerated from them on every push to `main`
(see [Packaging](../packaging/#documentation-site)).
