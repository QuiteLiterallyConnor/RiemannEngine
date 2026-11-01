---
title: Building from source
nav_order: 12
description: CMake options, Windows and Linux builds, Docker, CUDA architectures and the test suites.
permalink: /building/
---

# Building from source
{: .no_toc }

Most users should install a [release](../installation/). Build from source to target other GPU
architectures, other CUDA or LibTorch versions, CPU-only machines, or to work on RiemannEngine
itself.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## Requirements

| | Windows | Linux |
|---|---|---|
| Compiler | Visual Studio 2022 (MSVC 19.4x), C++20 | GCC 11+ or Clang 14+, C++20 |
| CMake | 3.24+ (bundled with VS 2022 works) | 3.24+ |
| Generator | Ninja (recommended) | Ninja or Make |
| CUDA | 13.2 or 13.4 (optional with `CPU_ONLY`) | 12.4, 13.2 or 13.4 |
| cuDNN | Matching the LibTorch build | Matching the LibTorch build |
| LibTorch | cxx11 ABI build matching your CUDA, e.g. `libtorch_cu134` | same |

{: .warning }
> Use the LibTorch build for **the same CUDA major/minor** as your toolkit. Mixing them, for
> example a cu132 LibTorch with nvcc 13.4, produces link errors or runtime crashes.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `CPU_ONLY` | `OFF` | Build without CUDA kernels. No CUDA toolkit needed |
| `CMAKE_CUDA_ARCHITECTURES` | `86` | GPU SM list, e.g. `"75;86;89;90"`. CUDA 13 needs SM 7.5 or newer |
| `RIEMANN_ENABLE_GAUSSIAN` | `ON` | Build `riemann::gaussian` (splat/ sources and headers) |
| `RIEMANN_ENABLE_OPENMP` | `ON` | OpenMP for CPU kernels (MSVC uses the LLVM runtime to avoid the LibTorch `libiomp5md` clash) |
| `RIEMANN_BUILD_TESTS` | `ON` | `engine_tests`, `cpu_convolution_tests`, `gaussian_tests`, `train` |
| `RIEMANN_BUILD_ADAPTIVE_TESTS` | `ON` | `adaptive_tests` and the adaptive benchmarks (`adaptive_bench`, `voxelize_obj_bench`, `sphere`) |
| `RIEMANN_BUILD_GAUSSIAN_BENCHMARKS` | `ON` | `gaussian_bench`, `gaussian_ply_bench` |
| `RIEMANN_USE_CCACHE` | `ON` | Use ccache or sccache if found |
| `RIEMANN_COPY_TORCH_DLLS` | `ON` | Windows: copy LibTorch DLLs next to the built executables |

The library is the static target `riemann_cpp`, exported as `RiemannEngine::RiemannEngine` (and
`RiemannEngine::riemann_cpp`).

## Linux

```bash
git clone https://github.com/QuiteLiterallyConnor/RiemannEngine.git
cd RiemannEngine
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_CUDA_ARCHITECTURES="86;89"
cmake --build build -j
sudo cmake --install build --prefix /usr/local/RiemannEngine/1.0.0_cu134
```

The installed CMake package is **relocatable**: copy the prefix anywhere and point
`CMAKE_PREFIX_PATH` at it.

## Windows

### With the scripts

```powershell
# Every Windows + Linux variant (Linux via Docker)
.\scripts\build-all-windows.ps1 -Clean

# Only Windows cu134, installed to C:\sdk\RiemannEngine\1.0.0_cu134
.\scripts\build-all-windows.ps1 -CudaVersion 134 -SkipLinux -Install -InstallPrefix C:\sdk

# CPU only
.\scripts\build-all-windows.ps1 -CpuOnly -Clean
```

`build-all-windows.ps1` enters the VS 2022 developer environment by itself. It expects LibTorch in
`<LibTorchRoot>\libtorch_cu<X>` (default root `D:\`) and writes each variant to
`<BuildDir>-cu<X>` (default `build-cu132`, `build-cu134`).

| Parameter | Meaning |
|---|---|
| `-CudaVersion` | Optional filter, e.g. `134` or `124,132`. Default: every variant |
| `-SkipWindows`, `-SkipLinux` | Skip a platform |
| `-LibTorchRoot`, `-LibTorchPath`, `-CudaPath`, `-CudnnPath` | Dependency locations |
| `-Install`, `-InstallPrefix` | Install each Windows variant to `<prefix>\RiemannEngine\<ver>_cu<X>` |
| `-Jobs`, `-BuildType`, `-Clean`, `-Verbose`, `-CudaArchitectures` | Build controls |

### By hand

From an *x64 Native Tools* prompt:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_PREFIX_PATH=D:\libtorch_cu134 ^
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
cmake --install build --prefix C:\sdk\RiemannEngine\1.0.0_cu134
```

`scripts\configure-windows.ps1` and `scripts\build-windows.ps1` wrap these two steps with
parameters for the LibTorch, CUDA and Ninja paths.

## Docker

```powershell
.\docker\docker-build.ps1 -CudaVersion 134 -Tag dev       # local image
```

```bash
./docker/docker-build.sh --cuda-version 134 --tag dev
```

The image builds RiemannEngine inside an `nvidia/cuda:<ver>-devel-ubuntu<U>` base
(`12.4.0` on 22.04, `13.2.1-cudnn` and `13.4.3-cudnn` on 24.04) with the matching LibTorch, and
installs it to `/usr/local`. `TORCH_CUDA_ARCH_LIST` is derived from the requested SM
numbers. See [Packaging](../packaging/) for publishing.

## Running the tests

```bash
ctest --test-dir build --output-on-failure     # all suites
./build/engine_tests                            # high-level engine tests
./build/adaptive_tests
./build/gaussian_tests
./build/cpu_convolution_tests
```

| Suite | Covers |
|---|---|
| `engine_tests` | Batching, every layer family, residual blocks, training loops, persistence |
| `cpu_convolution_tests` | Convolution kernels against reference implementations |
| `adaptive_tests` | Geometry, topology, numerics, gradients, regression, sphere and OBJ voxelization |
| `gaussian_tests` | Containers, transforms, rendering (CPU/CUDA parity), PLY, training |

Tests that need data look in `tests/data`, a path compiled in as `RIEMANN_TEST_DATA_DIR`. The
Gaussian PLY and render tests also accept a `RIEMANN_TEST_DATA_DIR` environment variable. Tests
that need a GPU or a data file **skip** rather than fail when it is missing.

The release builders have their own tests:

```powershell
.\tests\builders\test-builders.ps1
```

```bash
bash tests/builders/test-builders.sh
```

## Source layout

```
RiemannEngine/
  riemann.hpp          umbrella header
  engine/              uniform sparse engine: tensor/, nn/, grid/, ops/, cuda/, learning/, helpers/, io/
  octree/              adaptive voxelization (riemann::adaptive)
  splat/               Gaussian splats (riemann::gaussian)
  autodiff/            autograd Functions
  core/ layers/ dynamic/ normal/ helpers/ learning/   forwarding headers kept for compatibility
  vendor/              bundled third-party headers
tests/                 engine/, adaptive/, gaussian/, convolution/, builders/
scripts/ apt/ docker/ msi/     build and release tooling
docs/                  this site (Jekyll + Doxygen)
```
