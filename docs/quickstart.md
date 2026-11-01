---
title: Quick start
nav_order: 3
description: Write, build and run your first RiemannEngine program.
permalink: /quickstart/
---

# Quick start
{: .no_toc }

From an installed SDK to a sparse network running on the GPU in about five minutes. This page
assumes you finished [Installation](../installation/) for one variant. The examples use `cu134`.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## 1. Write the program

Create `hello_riemann/main.cpp`:

```cpp
#include <riemann.hpp>

#include <iostream>

int main() {
    torch::manual_seed(0);
    const torch::Device device =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "RiemannEngine " << riemann::VERSION << " on " << device << "\n";

    // Two small point clouds on an integer 3D grid (duplicates removed).
    auto cloud_a = std::get<0>(torch::unique_dim(
        torch::randint(0, 16, {500, 3}, torch::kInt32), 0));
    auto cloud_b = std::get<0>(torch::unique_dim(
        torch::randint(0, 16, {300, 3}, torch::kInt32), 0));

    // Prepend a batch index to each cloud and stack them: [N, 1 + 3].
    auto coords = riemann::batched_coordinates({cloud_a, cloud_b},
                                               torch::kInt32, device);
    auto features = torch::rand({coords.size(0), 3}, device);
    riemann::SparseTensor input(features, coords);

    // conv(3 -> 16) -> BN -> ReLU -> global average pool -> linear(16 -> 4)
    riemann::RiemannConvolution conv(3, 16, /*kernel_size=*/3, /*stride=*/1,
                                     /*dilation=*/1, /*bias=*/true,
                                     /*expand_coordinates=*/false,
                                     riemann::ConvolutionMode::DEFAULT,
                                     /*dimension=*/3);
    riemann::RiemannBatchNorm norm(16);
    riemann::RiemannReLU relu;
    riemann::RiemannGlobalAvgPooling pool;
    riemann::RiemannLinear head(16, 4);
    conv->to(device);
    norm->to(device);
    head->to(device);

    auto x = relu->forward(norm->forward(conv->forward(input)));
    auto logits = head->forward(pool->forward(x));

    std::cout << "input voxels : " << input.F().size(0) << "\n"
              << "logits shape : " << logits.F().sizes() << "\n"
              << logits.F() << "\n";
    return 0;
}
```

### What this does

| Line | Meaning |
|---|---|
| `batched_coordinates({a, b})` | Prepends a batch index column, giving an `[N, 4]` int32 tensor `(b, x, y, z)`. See [coordinates](../concepts/#coordinates). |
| `SparseTensor input(features, coords)` | Features `[N, 3]` paired with coordinates. A fresh [coordinate manager](../concepts/#the-coordinate-manager) is created and the coordinates are hashed. |
| `RiemannConvolution(3, 16, 3, ...)` | A 3×3×3 sparse convolution. With stride 1 the output keeps the input coordinates. |
| `RiemannGlobalAvgPooling` | Averages each batch item, giving one row per batch: `[2, 16]`. |
| `RiemannLinear` | A per-row linear layer that preserves coordinates. |

Modules are LibTorch `torch::nn` module holders: call `->forward(...)`, `->to(device)`,
`->parameters()`, `->train()` and `->eval()` exactly as you would with LibTorch.

## 2. Build it

### Option A: CMake

`hello_riemann/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(hello_riemann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(RiemannEngine 1.0 CONFIG REQUIRED)

add_executable(hello_riemann main.cpp)
target_link_libraries(hello_riemann PRIVATE RiemannEngine::RiemannEngine)

# Windows: copy the LibTorch DLLs next to the executable.
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET hello_riemann POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS} $<TARGET_FILE_DIR:hello_riemann>)
endif()
```

`RiemannEngine::RiemannEngine` carries the include paths, compile definitions (`USE_CUDA`,
`RIEMANN_ENABLE_GAUSSIAN`, ...), LibTorch, the CUDA libraries and OpenMP.

**Windows** (*x64 Native Tools Command Prompt for VS 2022*):

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      "-DCMAKE_PREFIX_PATH=C:/Program Files/RiemannEngine/1.0.0_cu134;C:/libtorch_cu134"
cmake --build build
build\hello_riemann.exe
```

**Ubuntu:**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="/usr/local/RiemannEngine/1.0.0_cu134;/opt/libtorch"
cmake --build build
./build/hello_riemann
```

{: .warning }
> On Windows always build `Release` or `RelWithDebInfo`. The library and LibTorch use the release
> CRT, so a Debug build fails to link.

### Option B: g++ (Ubuntu)

```bash
RIEMANN=/usr/local/RiemannEngine/1.0.0_cu134
LIBTORCH=/opt/libtorch
CUDA=/usr/local/cuda

g++ -std=c++20 -O2 -fopenmp main.cpp -o hello_riemann \
    -DRIEMANN_NO_PYBIND -DUSE_CUDA -DRIEMANN_ENABLE_GAUSSIAN \
    -I$RIEMANN/include/riemann -I$RIEMANN/include/riemann/vendor \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include \
    -I$CUDA/include \
    -L$RIEMANN/lib -L$LIBTORCH/lib -L$CUDA/lib64 \
    -lriemann_cpp \
    -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed \
    -ltorch -ltorch_cpu -lc10 -lcudart -lcublas -lcusparse \
    -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUDA/lib64
```

`--no-as-needed` keeps `libtorch_cuda` linked even though nothing references it directly.
Without it, LibTorch reports that no CUDA backend is available at run time.

### Option C: Makefile (Ubuntu)

```make
RIEMANN  ?= /usr/local/RiemannEngine/1.0.0_cu134
LIBTORCH ?= /opt/libtorch
CUDA     ?= /usr/local/cuda

CXX      := g++
CXXFLAGS := -std=c++20 -O2 -fopenmp \
            -DRIEMANN_NO_PYBIND -DUSE_CUDA -DRIEMANN_ENABLE_GAUSSIAN \
            -I$(RIEMANN)/include/riemann -I$(RIEMANN)/include/riemann/vendor \
            -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include \
            -I$(CUDA)/include
LDFLAGS  := -fopenmp -L$(RIEMANN)/lib -L$(LIBTORCH)/lib -L$(CUDA)/lib64 \
            -Wl,-rpath,$(LIBTORCH)/lib -Wl,-rpath,$(CUDA)/lib64
LDLIBS   := -lriemann_cpp \
            -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed \
            -ltorch -ltorch_cpu -lc10 -lcudart -lcublas -lcusparse

hello_riemann: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f hello_riemann

.PHONY: clean
```

In the Docker image, run `make RIEMANN=/usr/local`.

## 3. Expected output

```
RiemannEngine 1.0.0 on cuda
input voxels : 761
logits shape : [2, 4]
 0.2063  0.0851 -0.4398 -0.0131
 0.2050  0.0237 -0.3335 -0.0289
[ CUDAFloatType{2,4} ]
```

If you see `on cpu`, LibTorch could not find a CUDA device. See
[Troubleshooting](../troubleshooting/#the-program-runs-on-cpu).

## Next steps

- [Core concepts](../concepts/): how coordinates, strides and maps work.
- [Sparse tensors & layers](../sparse-tensors/): every layer and operation.
- [Training](../training/): add a loss and an optimizer.
- [Examples](../examples/): complete classification and segmentation networks.
