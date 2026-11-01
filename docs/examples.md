---
title: Examples
nav_order: 11
description: Complete, compilable programs - classification, U-Net segmentation, adaptive voxelization and Gaussian splat fitting.
permalink: /examples/
---

# Examples
{: .no_toc }

Four complete programs. Each one compiles against an installed release and was run on CUDA 13.4.
Every example falls back to the CPU when no GPU is present.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## Building the examples

Put the sources next to this `CMakeLists.txt` and build as in the [Quick start](../quickstart/):

```cmake
cmake_minimum_required(VERSION 3.24)
project(riemann_examples LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(RiemannEngine REQUIRED)
foreach(ex classify segment adaptive_demo splat_demo)
  add_executable(${ex} ${ex}.cpp)
  target_link_libraries(${ex} PRIVATE RiemannEngine::RiemannEngine)
endforeach()
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="<riemann prefix>;<libtorch>"
cmake --build build --config Release
```

## 1. Shape classification

Classifies voxelized point clouds as **sphere** or **cube**. It shows `sparse_quantize`,
`batched_coordinates`, a strided encoder, global pooling as the classifier head,
`cross_entropy_sparse` and `save_model`.

```cpp
// Classify voxelized point clouds as sphere (0) or cube (1).
#include <riemann.hpp>
#include <iostream>

using namespace riemann;
namespace tr = riemann::training;

// Random points on the surface of a unit sphere or cube, scaled and quantized.
at::Tensor make_shape(int label, int64_t n = 2048) {
    auto p = torch::randn({n, 3});
    if (label == 0) {
        p = p / p.norm(2, 1, true);                        // sphere surface
    } else {
        p = p / std::get<0>(p.abs().max(1, true));          // cube surface
    }
    const double scale = 0.6 + 0.4 * torch::rand({1}).item<double>();
    auto q = sparse_quantize(p * scale, c10::nullopt, c10::nullopt, -100, /*voxel=*/0.05);
    return q.coordinates;                                   // [M, 3] int voxel coordinates
}

struct NetImpl : torch::nn::Module {
    RiemannConvolution c1{nullptr}, c2{nullptr}, c3{nullptr};
    RiemannBatchNorm b1{nullptr}, b2{nullptr}, b3{nullptr};
    RiemannReLU relu;
    RiemannGlobalMaxPooling pool;
    RiemannLinear head{nullptr};

    NetImpl(int in, int classes) {
        c1 = register_module("c1", RiemannConvolution(in, 16, 3, 1));
        b1 = register_module("b1", RiemannBatchNorm(16));
        c2 = register_module("c2", RiemannConvolution(16, 32, 2, 2));
        b2 = register_module("b2", RiemannBatchNorm(32));
        c3 = register_module("c3", RiemannConvolution(32, 64, 2, 2));
        b3 = register_module("b3", RiemannBatchNorm(64));
        head = register_module("head", RiemannLinear(64, classes));
        register_module("relu", relu);
        register_module("pool", pool);
    }
    SparseTensor forward(const SparseTensor& x) {
        auto y = relu->forward(b1->forward(c1->forward(x)));
        y = relu->forward(b2->forward(c2->forward(y)));
        y = relu->forward(b3->forward(c3->forward(y)));
        return head->forward(pool->forward(y));             // [B, classes]
    }
};
TORCH_MODULE(Net);

int main() {
    torch::manual_seed(0);
    const torch::Device dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    Net net(1, 2);
    net->to(dev);
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(1e-2));

    const int batch = 8;
    for (int it = 1; it <= 60; ++it) {
        std::vector<at::Tensor> clouds;
        auto labels = torch::randint(0, 2, {batch}, torch::kInt64);
        for (int b = 0; b < batch; ++b) clouds.push_back(make_shape(labels[b].item<int>()));

        auto coords = batched_coordinates(clouds, torch::kInt32, dev);
        SparseTensor x(torch::ones({coords.size(0), 1}, dev), coords);

        net->train();
        auto logits = net->forward(x);
        auto loss = tr::cross_entropy_sparse(logits, labels.to(dev));
        opt.zero_grad();
        loss.backward();
        opt.step();

        if (it % 10 == 0) {
            auto acc = (logits.F().argmax(1).cpu() == labels).to(torch::kFloat).mean();
            std::cout << "iter " << it << "  loss " << loss.item<double>()
                      << "  batch acc " << acc.item<double>() << "\n";
        }
    }
    save_model(net, "classifier.pt");
}
```

```text
iter 10  loss 0.0533567  batch acc 1
...
iter 60  loss 0.00233173  batch acc 1
```

{: .note }
> `RiemannGlobalMaxPooling` returns one row per batch index, in batch order, so the logits line up
> with `labels` directly.

## 2. Per-point segmentation with a U-Net

A two-level sparse U-Net predicts a label for every **input point**. Not every voxel needs one.
It shows:

- `TensorField` for continuous coordinates, averaged into voxels by `field.sparse()`.
- Strided `RiemannConvolution` down and `RiemannConvolutionTranspose` up. The transpose
  returns to the **same coordinate map** as the encoder at that stride, so `riemann::cat` skip
  connections work.
- `field.inverse_mapping(key)` to map voxel logits back to points.

```cpp
// Per-point segmentation with a small sparse U-Net and a TensorField.
// Task: label each point of a noisy scene as inside (1) or outside (0) a ball.
#include <riemann.hpp>
#include <iostream>

using namespace riemann;

struct ConvBNReLUImpl : torch::nn::Module {
    RiemannConvolution conv{nullptr};
    RiemannBatchNorm bn{nullptr};
    RiemannReLU relu;
    ConvBNReLUImpl(int in, int out, int k = 3, int stride = 1) {
        conv = register_module("conv", RiemannConvolution(in, out, k, stride));
        bn = register_module("bn", RiemannBatchNorm(out));
    }
    SparseTensor forward(const SparseTensor& x) {
        return relu->forward(bn->forward(conv->forward(x)));
    }
};
TORCH_MODULE(ConvBNReLU);

struct UNetImpl : torch::nn::Module {
    ConvBNReLU enc1{nullptr}, down1{nullptr}, enc2{nullptr}, down2{nullptr}, bottom{nullptr};
    RiemannConvolutionTranspose up2{nullptr}, up1{nullptr};
    ConvBNReLU dec2{nullptr}, dec1{nullptr};
    RiemannConvolution head{nullptr};

    UNetImpl(int in, int classes) {
        enc1   = register_module("enc1", ConvBNReLU(in, 16));
        down1  = register_module("down1", ConvBNReLU(16, 32, 2, 2));   // stride 2
        enc2   = register_module("enc2", ConvBNReLU(32, 32));
        down2  = register_module("down2", ConvBNReLU(32, 64, 2, 2));   // stride 4
        bottom = register_module("bottom", ConvBNReLU(64, 64));
        up2    = register_module("up2", RiemannConvolutionTranspose(64, 32, 2, 2));
        dec2   = register_module("dec2", ConvBNReLU(64, 32));          // 32 skip + 32 up
        up1    = register_module("up1", RiemannConvolutionTranspose(32, 16, 2, 2));
        dec1   = register_module("dec1", ConvBNReLU(32, 16));          // 16 skip + 16 up
        head   = register_module("head", RiemannConvolution(16, classes, 1, 1, 1, /*bias=*/true));
    }
    SparseTensor forward(const SparseTensor& x) {
        auto s1 = enc1->forward(x);                       // stride 1
        auto s2 = enc2->forward(down1->forward(s1));      // stride 2
        auto b  = bottom->forward(down2->forward(s2));    // stride 4
        auto u2 = dec2->forward(cat({up2->forward(b), s2}));   // back on s2's coordinates
        auto u1 = dec1->forward(cat({up1->forward(u2), s1}));  // back on s1's coordinates
        return head->forward(u1);
    }
};
TORCH_MODULE(UNet);

int main() {
    torch::manual_seed(0);
    const torch::Device dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    const double voxel = 0.05;

    UNet net(3, 2);
    net->to(dev);
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(5e-3));

    for (int it = 1; it <= 100; ++it) {
        // Scene: 20k points in a [-1, 1]^3 box; label = inside a ball of random radius.
        auto pts = torch::rand({20000, 3}) * 2 - 1;
        const double r = 0.4 + 0.3 * torch::rand({1}).item<double>();
        auto labels = (pts.norm(2, 1) < r).to(torch::kInt64).to(dev);

        // TensorField coordinates: [N, 1 + 3] float in voxel units, column 0 = batch.
        auto coords = torch::cat({torch::zeros({pts.size(0), 1}), pts / voxel}, 1).to(dev);
        TensorField field(pts.to(dev), coords);              // features = xyz
        SparseTensor voxels = field.sparse();                // averaged per voxel

        net->train();
        auto out = net->forward(voxels);                      // same coordinates as `voxels`
        auto per_point = out.F().index_select(
            0, field.inverse_mapping(out.coordinate_map_key()).to(torch::kLong));
        auto loss = torch::nn::functional::cross_entropy(per_point, labels);

        opt.zero_grad();
        loss.backward();
        opt.step();

        if (it % 20 == 0) {
            auto acc = (per_point.argmax(1) == labels).to(torch::kFloat).mean();
            std::cout << "iter " << it << "  voxels " << voxels.F().size(0) << "  loss "
                      << loss.item<double>() << "  point acc " << acc.item<double>() << "\n";
        }
    }
}
```

```text
iter 20  voxels 17165  loss 0.191532  point acc 0.9574
...
iter 100  voxels 17160  loss 0.0784989  point acc 0.9665
```

## 3. Adaptive voxelization

Builds an adaptive tensor from a clustered point cloud and refines where the points are dense. It
then runs a differentiable adaptive convolution, pools, and voxelizes a sphere to 99 % volume
accuracy. See [Adaptive voxelization](../adaptive/).

```cpp
// Adaptive voxelization: refine where points are dense, then convolve and pool.
#include <riemann.hpp>
#include <octree/adaptive.hpp>
#include <iostream>

using namespace riemann::adaptive;

int main() {
    torch::manual_seed(0);

    // A dense cluster near the origin plus sparse background points in [0, 16)^3.
    auto cluster = torch::randn({4000, 3}, torch::kFloat64) * 0.8 + 2.0;
    auto background = torch::rand({500, 3}, torch::kFloat64) * 16.0;
    auto points = torch::cat({cluster, background}).clamp(0.0, 15.999);
    auto features = torch::ones({points.size(0), 1}, torch::kFloat64);

    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {4.0, 4.0, 4.0};
    cfg.min_voxel_size  = {0.5, 0.5, 0.5};
    cfg.max_depth       = 3;                     // 4 -> 2 -> 1 -> 0.5

    auto t = AdaptiveSparseTensor::from_points(points, features, cfg);
    std::cout << "level-0 leaves: " << t.size() << "\n";

    t.refine_by_density(/*points_per_leaf_threshold=*/32);
    t.revoxelize_from_points();                  // recompute features on the new leaves
    auto counts = t.counts_by_level();
    std::cout << "after refine:   " << t.size() << " leaves  (per level:";
    for (auto c : counts) std::cout << " " << c;
    std::cout << ")  overlapping=" << t.has_overlapping_leaves() << "\n";

    // 3x3x3 adaptive convolution, 1 -> 8 channels, then a volume-weighted global average.
    AdaptiveKernelConfig kc;
    kc.kernel_size = 3;
    kc.mode = AdaptiveConvMode::VolumeWeighted;
    auto weight = torch::randn({27, 1, 8}, torch::kFloat64).mul_(0.1).requires_grad_(true);

    auto y = adaptive_convolution(t, weight, kc);
    auto pooled = adaptive_global_avg_pool(y);   // [8]
    pooled.sum().backward();
    std::cout << "conv out: " << y.features().sizes() << "  pooled: " << pooled.sizes()
              << "  |grad W| = " << weight.grad().norm().item<double>() << "\n";

    auto stats = KernelMapCache::instance().stats();
    std::cout << "kernel-map cache: " << stats.entries << " entries, " << stats.misses
              << " misses\n";

    // Exact sphere voxelization to 99% volume accuracy.
    double achieved = 0.0;
    auto sphere = voxelize_sphere(1.0, 99.0, {0, 0, 0}, 10'000'000, 10, &achieved);
    std::cout << "sphere: " << sphere.size() << " leaves, accuracy " << achieved << "%\n";
}
```

```text
level-0 leaves: 64
after refine:   358 leaves  (per level: 62 8 32 256)  overlapping=0
conv out: [358, 8]  pooled: [8]  |grad W| = 13.7532
kernel-map cache: 1 entries, 1 misses
sphere: 204 leaves, accuracy 99.2232%
```

## 4. Fitting Gaussian splats

Renders eight views of a reference splat scene, then trains a second model from a different random
initialization with `GaussianTrainer`, including densification and pruning. The result is written
as a standard 3DGS PLY. See [Gaussian splats](../gaussian/).

```cpp
// Fit Gaussian splats to rendered views of a reference scene, then save a PLY.
#include <riemann.hpp>
#include <cmath>
#include <iostream>

using namespace riemann::gaussian;

int main() {
    torch::manual_seed(0);
    const torch::Device dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Reference scene: 200 coloured Gaussians in a ball.
    auto ref_pts = torch::randn({200, 3}) * 0.4;
    auto ref_rgb = torch::rand({200, 3});
    GaussianModel truth = GaussianModel::from_points(ref_pts, ref_rgb, /*sh_degree=*/0, 0.8).to(dev);

    // Eight cameras on a ring, looking at the origin (y-up world -> up = {0,-1,0}).
    std::vector<std::pair<Camera, at::Tensor>> views;
    for (int i = 0; i < 8; ++i) {
        const double a = 2.0 * 3.14159265358979 * i / 8;
        auto cam = Camera::look_at({3 * std::sin(a), 0.5, -3 * std::cos(a)}, {0, 0, 0},
                                   {0, -1, 0}, 128, 128, /*fov_x=*/0.9);
        torch::NoGradGuard ng;
        views.emplace_back(cam, render(truth, cam).image.clone());
    }

    // Start from a different random cloud and train with the 3DGS recipe.
    GaussianModel model = GaussianModel::from_points(torch::randn({300, 3}) * 0.5,
                                                     torch::full({300, 3}, 0.5), 0, 0.1).to(dev);
    GaussianTrainerConfig cfg;
    cfg.scene_extent = 1.0;
    cfg.densify_start = 100;
    cfg.densify_stop = 400;
    GaussianTrainer trainer(model, cfg);

    for (int it = 1; it <= 600; ++it) {
        const auto& [cam, target] = views[it % views.size()];
        auto r = trainer.step(cam, target);
        if (it % 100 == 0)
            std::cout << "iter " << it << "  loss " << r.loss << "  gaussians " << r.gaussians
                      << "\n";
    }

    save_gaussian_ply(model, "fitted.ply");
    auto reloaded = load_gaussian_model_ply("fitted.ply");
    std::cout << "saved fitted.ply with " << reloaded.size() << " Gaussians\n";
}
```

```text
iter 100  loss 0.091194  gaussians 300
iter 200  loss 0.0694274  gaussians 184
...
iter 600  loss 0.00896243  gaussians 214
saved fitted.ply with 214 Gaussians
```

Open `fitted.ply` in any 3DGS viewer, for example SuperSplat.

## More

The test suites are also good references:

| Source | Shows |
|---|---|
| [`tests/engine/test_pipeline.cpp`](https://github.com/QuiteLiterallyConnor/RiemannEngine/blob/main/tests/engine/test_pipeline.cpp) | Training loop, accumulation, scheduler, persistence, decomposition |
| [`tests/engine/test_layers.cpp`](https://github.com/QuiteLiterallyConnor/RiemannEngine/blob/main/tests/engine/test_layers.cpp) | Every layer family, residual blocks, union and structural ops |
| [`tests/adaptive/`](https://github.com/QuiteLiterallyConnor/RiemannEngine/tree/main/tests/adaptive) | Topology, refinement, kernels, sphere voxelization |
| [`tests/gaussian/`](https://github.com/QuiteLiterallyConnor/RiemannEngine/tree/main/tests/gaussian) | Rendering, PLY round-trips, training, CUDA vs. CPU parity |
