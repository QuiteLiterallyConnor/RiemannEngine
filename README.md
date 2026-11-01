<div align="center">

# RiemannEngine

**CUDA-accelerated sparse tensor engine for native C++, with adaptive mixed-resolution voxelization and 3D Gaussian splats.**

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-acaicia%2Friemannengine-0db7ed?logo=docker&logoColor=white&style=for-the-badge)](https://hub.docker.com/r/acaicia/riemannengine)
[![Documentation](https://img.shields.io/badge/Docs-GitHub%20Pages-222222?logo=github&logoColor=white&style=for-the-badge)](https://quiteliterallyconnor.github.io/RiemannEngine/)
[![Releases](https://img.shields.io/badge/Download-Releases-2ea44f?logo=github&logoColor=white&style=for-the-badge)](https://github.com/QuiteLiterallyConnor/RiemannEngine/releases)

</div>

## Features

### Sparse tensors and sparse convolutional networks
- `SparseTensor` and `TensorField` hold LibTorch features with integer coordinates, in any spatial dimension (3D, 4D spatio-temporal, …).
- A coordinate manager and kernel-map generation, cached and reused across layers.
- Sparse convolution, transposed convolution and generative transposed convolution.
- Max, average, sum and global pooling; pooling transpose, interpolation, pruning, union and broadcast ops.
- Batch, instance, group and layer normalization; ReLU, ELU, GELU, SiLU and 20+ other activations; linear layers and dropout.
- Building blocks: residual `BasicBlock`/`Bottleneck` and squeeze-excitation blocks.
- `TORCH_MODULE` layers (`RiemannConvolution`, `RiemannBatchNorm`, `RiemannGlobalAvgPooling`, …) that work with LibTorch autograd and optimizers.
- Batching helpers: `batched_coordinates`, `sparse_collate`, `SparseCollation`.
- Training utilities: sparse losses, optimizer/scheduler factories, gradient accumulation, and model and tensor persistence.

### Adaptive (mixed-resolution) voxelization
- A single sparse tensor can hold voxels at **different resolutions** (octree-style, up to a configurable depth).
- Refine only the regions that matter (a region of interest, dense or high-variance areas) and keep the rest coarse.
- Convolution, pooling, interpolation and resampling all work **across resolution boundaries**, are differentiable, and run on CPU and CUDA.
- In the included benchmark the adaptive grid uses **41,376 voxels**, where the equivalent uniform grid needs about **35 million**.
- Opt-in: the uniform `SparseTensor` API is unchanged.

### 3D Gaussian splats
- Loads 3DGS `.ply` files, both standard and SuperSplat/PlayCanvas compressed (detected automatically), into a batched, LibTorch-backed `GaussianTensor`.
- Covariances, opacity and bounds filtering, selection, merging, rigid transforms, and spherical harmonics up to degree 3.
- **Differentiable rasterizer** with fused CUDA forward/backward kernels, plus a LibTorch fallback on CPU.
- A 3DGS-style `GaussianTrainer` with densification, pruning and opacity reset, and PLY export.
- On an RTX 3080, a 139K-Gaussian scene loads in about 17–37 ms and a 512×512 render takes about 6 ms.

### Native, GPU-first C++
- C++20 static library (`riemann_cpp`) with a relocatable CMake package (`find_package(RiemannEngine)`).
- CUDA kernels backed by cuBLAS and cuSPARSE, plus a CPU path with OpenMP.
- Prebuilt MSI, `.deb`, SDK archives and Docker images for each supported CUDA version.
- Optional [Python bindings](#python-bindings) (`riemannengine`) that interoperate with PyTorch tensors, autograd and optimizers.

---

## Platform Support

| OS | CUDA | LibTorch | Compiler | Release assets |
|---|---|---|---|---|
| Windows 10/11 x64 | 13.2 | 2.15.0 (cu132) | Visual Studio 2022 | `.msi`, `-win-x64.zip` |
| Windows 10/11 x64 | 13.4 | 2.15.0 (cu134) | Visual Studio 2022 | `.msi`, `-win-x64.zip` |
| Ubuntu 22.04 x64 | 12.4 | 2.5.1 (cu124, CXX11 ABI) | GCC 11 | `.deb`, `.tar.gz`, Docker `cu124` |
| Ubuntu 24.04 x64 | 13.2 | 2.15.0 (cu132) | GCC 13 | `.deb`, `.tar.gz`, Docker `cu132` |
| Ubuntu 24.04 x64 | 13.4 | 2.15.0 (cu134) | GCC 13 | `.deb`, `.tar.gz`, Docker `cu134` / `latest` |

- **GPU:** an NVIDIA GPU with a driver that supports your CUDA version. CUDA 13.x requires compute capability 7.5 (Turing) or newer. Release binaries are compiled for compute capability 8.6 (RTX 30-series); for other GPUs, [build from source](#building-from-source) with your architecture.
- **Side by side:** each variant installs into its own `RiemannEngine/<version>_cu<XXX>` folder, so several CUDA variants can be installed at once.
- **LibTorch must match:** use the LibTorch build listed for your variant. RiemannEngine links against it and does not bundle it.

---

## Quick Start

The steps below take you from nothing to a running RiemannEngine program. Pick one CUDA variant
(this guide uses **CUDA 13.4**, i.e. `cu134`) and use it for every step.

### Step 1: Download a release

Go to the **[Releases page](https://github.com/QuiteLiterallyConnor/RiemannEngine/releases)** and download the installer for your platform:

| Platform | Download | Installs to |
|---|---|---|
| Windows | `RiemannEngine-1.0.0-cu134-x64.msi` | `C:\Program Files\RiemannEngine\1.0.0_cu134\` |
| Windows (no installer) | `RiemannEngine-1.0.0-cu134-win-x64.zip` | wherever you extract it |
| Ubuntu 24.04 | `libriemann-cpp-dev_1.0.0_cu134_ubuntu24.04_amd64.deb` | `/usr/local/RiemannEngine/1.0.0_cu134/` |
| Ubuntu (no package manager) | `RiemannEngine-1.0.0-cu134-ubuntu24.04-x64.tar.gz` | wherever you extract it |

Every variant has the same layout:

```
RiemannEngine/1.0.0_cu134/
├── include/riemann/            # headers: #include <riemann.hpp>
└── lib/
    ├── riemann_cpp.lib         # Windows   (libriemann_cpp.a on Linux)
    └── cmake/RiemannEngine/    # CMake package: find_package(RiemannEngine)
```

### Step 2: Install the binary

<details open>
<summary><b>Windows</b></summary>

Double-click the `.msi`, or run it from a terminal:

```powershell
msiexec /i RiemannEngine-1.0.0-cu134-x64.msi
```

The installer also registers the package with CMake. With the zip, extract it anywhere, for example to `C:\` so the SDK lands in `C:\RiemannEngine\1.0.0_cu134`.

</details>

<details open>
<summary><b>Ubuntu</b></summary>

```bash
sudo apt install ./libriemann-cpp-dev_1.0.0_cu134_ubuntu24.04_amd64.deb
```

With the tarball, extract it instead: `sudo tar -xzf RiemannEngine-1.0.0-cu134-ubuntu24.04-x64.tar.gz -C /usr/local`.

</details>

### Step 3: Install the prerequisites

RiemannEngine is a static library. Your program also links against **CUDA** and **LibTorch**, so install the versions that match your variant.

<details open>
<summary><b>Windows</b></summary>

1. **NVIDIA driver:** install the latest [GeForce/RTX driver](https://www.nvidia.com/Download/index.aspx).
2. **CUDA Toolkit 13.4:** download it from the [CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive), choosing the version that matches your variant: 13.2 or 13.4. The installer sets `CUDA_PATH`.
3. **cuDNN 9:** download it from [cuDNN downloads](https://developer.nvidia.com/cudnn-downloads), choosing *Windows / x86_64 / CUDA 13*, and install it.
4. **Visual Studio 2022** with the **Desktop development with C++** workload ([download](https://visualstudio.microsoft.com/vs/older-downloads/)). CUDA 13 supports the VS 2022 toolset.
5. **CMake 3.24+** and **Ninja:**
   ```powershell
   winget install Kitware.CMake Ninja-build.Ninja
   ```
6. **LibTorch 2.15.0 for CUDA 13.4:** download the Release build and extract it, for example to `C:\libtorch_cu134`:
   ```powershell
   curl.exe -L -o libtorch.zip "https://download.pytorch.org/libtorch/cu134/libtorch-win-shared-with-deps-2.15.0%2Bcu134.zip"
   tar -xf libtorch.zip -C C:\
   Rename-Item C:\libtorch C:\libtorch_cu134
   ```
   For the cu132 variant, replace both `cu134` occurrences with `cu132`.
7. **Runtime DLLs:** add `C:\libtorch_cu134\lib` to your `PATH`, or copy its DLLs next to your executable. The CMake example below copies them for you.

</details>

<details open>
<summary><b>Ubuntu</b></summary>

1. **NVIDIA driver:**
   ```bash
   sudo ubuntu-drivers install
   sudo reboot
   nvidia-smi        # should list your GPU
   ```
2. **CUDA Toolkit and cuDNN** from NVIDIA's apt repository:
   ```bash
   # Ubuntu 24.04 (use ubuntu2204 for Ubuntu 22.04)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update

   sudo apt install cuda-toolkit-13-4 cudnn9-cuda-13   # cu134
   # sudo apt install cuda-toolkit-13-2 cudnn9-cuda-13 # cu132
   # sudo apt install cuda-toolkit-12-4 cudnn9-cuda-12 # cu124 (Ubuntu 22.04)

   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
3. **Build tools:**
   ```bash
   sudo apt install build-essential cmake ninja-build unzip wget libgomp1
   ```
4. **LibTorch:** install it to `/opt/libtorch`:
   ```bash
   # cu134 / cu132: LibTorch 2.15.0 (replace cu134 with cu132 in both places for cu132)
   wget -O libtorch.zip "https://download.pytorch.org/libtorch/cu134/libtorch-shared-with-deps-2.15.0%2Bcu134.zip"
   # cu124: LibTorch 2.5.1, CXX11 ABI
   # wget -O libtorch.zip "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip"

   sudo unzip -q libtorch.zip -d /opt
   ```

</details>

### Step 4: Write a program

Create a folder `hello_riemann/` containing `main.cpp`:

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

### Step 5: Compile it

Use whichever build method you prefer. **CMake works on both platforms** and is the recommended one.

#### Option A: CMake (Windows and Ubuntu)

`CMakeLists.txt`:

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

`RiemannEngine::RiemannEngine` brings in the include paths, compile definitions, LibTorch, CUDA and OpenMP, so nothing else is needed.

**Windows:** open *x64 Native Tools Command Prompt for VS 2022* and run:

```bat
cd hello_riemann
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      "-DCMAKE_PREFIX_PATH=C:/Program Files/RiemannEngine/1.0.0_cu134;C:/libtorch_cu134"
cmake --build build
```

> Always build `Release` (or `RelWithDebInfo`) on Windows. The library and LibTorch use the release CRT, and Debug builds fail to link.

**Ubuntu:**

```bash
cd hello_riemann
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="/usr/local/RiemannEngine/1.0.0_cu134;/opt/libtorch"
cmake --build build
```

#### Option B: g++ command (Ubuntu)

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

`--no-as-needed` keeps `libtorch_cuda` linked even though your code never calls it directly. Without it, LibTorch does not register its CUDA backend at runtime.

#### Option C: Makefile (Ubuntu)

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

```bash
make                     # or, for another variant: make RIEMANN=/usr/local/RiemannEngine/1.0.0_cu132
```

### Step 6: Run it

```bash
./build/hello_riemann          # CMake on Ubuntu (./hello_riemann for g++/make)
```
```bat
build\hello_riemann.exe        :: Windows
```

Expected output (the numbers will differ):

```
RiemannEngine 1.0.0 on cuda
input voxels : 761
logits shape : [2, 4]
 0.2063  0.0851 -0.4398 -0.0131
 0.2050  0.0237 -0.3335 -0.0289
[ CUDAFloatType{2,4} ]
```

If you see `on cpu`, LibTorch could not find a CUDA device. Check `nvidia-smi`, and make sure your LibTorch build matches your CUDA variant.

---

## Using the Docker Image

The [`acaicia/riemannengine`](https://hub.docker.com/r/acaicia/riemannengine) image is a ready-to-use development environment. It includes the CUDA toolkit, LibTorch, CMake, Ninja, and RiemannEngine prebuilt and installed. You don't need to install anything on the host except Docker and the NVIDIA driver.

| Tag | Ubuntu | CUDA | LibTorch |
|---|---|---|---|
| `cu134`, `latest`, `1.0.0-cu134` | 24.04 | 13.4.3 | 2.15.0 |
| `cu132`, `1.0.0-cu132` | 24.04 | 13.2.1 | 2.15.0 |
| `cu124`, `1.0.0-cu124` | 22.04 | 12.4.0 | 2.5.1 |

Inside the image:

| Path | Contents |
|---|---|
| `/usr/local/include/riemann` | RiemannEngine headers |
| `/usr/local/lib/libriemann_cpp.a` | RiemannEngine static library |
| `/usr/local/lib/cmake/RiemannEngine` | CMake package |
| `/opt/libtorch` | LibTorch |
| `/usr/local/cuda` | CUDA toolkit |
| `/workspace` | Working directory (mount your project here) |

### 1. Prepare the host

- **Linux:** install [Docker Engine](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then enable it:
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- **Windows:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL 2 backend and a current NVIDIA driver. GPU support works out of the box.

### 2. Pull and run

```bash
docker pull acaicia/riemannengine:cu134

# From your project folder (e.g. hello_riemann/ from the Quick Start)
docker run --gpus all --rm -it -v "$(pwd):/workspace" acaicia/riemannengine:cu134
```

In PowerShell, use `-v "${PWD}:/workspace"`. The flags:
- `--gpus all` exposes the host GPUs.
- `-v …:/workspace` mounts the current folder.
- `--rm` removes the container when you exit.

### 3. Build inside the container

The same `CMakeLists.txt` from [Step 5](#option-a-cmake-windows-and-ubuntu) works unchanged:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch
cmake --build build
./build/hello_riemann
```

For the g++ command or the Makefile, point `RIEMANN` at `/usr/local`:

```bash
make RIEMANN=/usr/local
```

Files written under `/workspace` persist on the host after the container exits.

For more (running as your own user, Docker Compose, VS Code Dev Containers, slim deployment images, supported GPUs and troubleshooting), see the [Docker guide](https://quiteliterallyconnor.github.io/RiemannEngine/docker/).

---

## Python Bindings

The `riemannengine` Python package exposes the full engine — sparse tensors and layers, adaptive voxelization and Gaussian splats — on top of PyTorch tensors and autograd. The extension compiles against the torch installed in your environment, so install torch first and build without isolation:

```bash
pip install torch                    # or a CUDA variant of torch
pip install scikit-build-core ninja
pip install --no-build-isolation .   # from the repository root
```

```python
import torch
import riemannengine as re

x = re.SparseTensor(features, coordinates)   # plain torch.Tensors in/out
conv = re.nn.RiemannConvolution(4, 8, kernel_size=3, dimension=3)
y = conv(x)
y.F.sum().backward()                         # autograd flows through
torch.optim.SGD(conv.parameters(), lr=0.1).step()
```

With a CPU-only torch the build automatically produces a CPU-only engine; with a CUDA torch the matching CUDA toolkit is required. See [python/README.md](python/README.md) for the full guide, including `riemannengine.adaptive` and `riemannengine.gaussian` examples.

---

## Building from Source

Prebuilt binaries cover the variants in [Platform Support](#platform-support). Build from source to target a different GPU architecture or CUDA/LibTorch combination.

- **Windows:** `.\scripts\build-all-windows.ps1` builds every Windows CUDA variant natively and every Linux variant through Docker. Add `-CudaVersion 134 -SkipLinux` for a single build. See [scripts/README.md](scripts/README.md).
- **Linux:** `./scripts/build-all-linux.sh --libtorch /opt/libtorch --install`.
- **Packages:** [msi/README.md](msi/README.md) (MSI), [apt/README.md](apt/README.md) (`.deb`), [docker/README.md](docker/README.md) (images), and `scripts\build-release.ps1`, which produces every release artifact at once.

## Documentation

Full API guides, adaptive voxelization and Gaussian splat walkthroughs, and troubleshooting are on the **[documentation site](https://quiteliterallyconnor.github.io/RiemannEngine/)**.
