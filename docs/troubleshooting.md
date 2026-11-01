---
title: Troubleshooting
nav_order: 14
description: Fixes for build, link, runtime and data problems, plus known limitations.
permalink: /troubleshooting/
---

# Troubleshooting
{: .no_toc }

1. TOC
{:toc}

---

## Configuring

### CMake cannot find Torch

Point `CMAKE_PREFIX_PATH` at the extracted **LibTorch root** (the folder that contains `lib/` and
`share/`), not at its `lib` folder. Separate multiple prefixes with `;` and quote the list:

```bash
cmake -B build -DCMAKE_PREFIX_PATH="/opt/libtorch;/usr/local/RiemannEngine/1.0.0_cu134"
```

Make sure you downloaded the build for the right OS, the cxx11 ABI (Linux) and the **same CUDA
version** as the RiemannEngine variant you installed.

### CMake cannot find RiemannEngine

- **MSI:** the installer registers the package, so `find_package(RiemannEngine)` should work. If
  you have several variants installed, CMake picks one, so pin it with
  `-DRiemannEngine_DIR="C:\Program Files\RiemannEngine\1.0.0_cu134\lib\cmake\RiemannEngine"`.
- **.deb, zip or tarball:** add the prefix (`/usr/local/RiemannEngine/<ver>_cu<X>` or wherever you
  extracted the archive) to `CMAKE_PREFIX_PATH`.
- Call `find_package(Torch REQUIRED)` **before** `find_package(RiemannEngine REQUIRED)`.

### `No CMAKE_CUDA_COMPILER could be found`

The CUDA toolkit isn't installed, or `nvcc` isn't on `PATH`. On Linux add
`/usr/local/cuda/bin` to `PATH`. On Windows reinstall the toolkit and open a new terminal.
Programs that only *use* RiemannEngine still need the toolkit, because LibTorch's CMake config
looks for it.

### Wrong CMake version or C++ standard

RiemannEngine needs CMake **3.24+** and **C++20**. Set `CMAKE_CXX_STANDARD 20` in your project.
The Docker builders for CUDA 13 use CMake 4.2.3.

### Unsupported GPU architecture

CUDA 13 cannot target GPUs older than SM 7.5 (Turing). Use the CUDA 12.4 variant for older cards.
When building from source, set both values:

```bash
-DCMAKE_CUDA_ARCHITECTURES=86      # and, for LibTorch's own flags:
export TORCH_CUDA_ARCH_LIST=8.6
```

LibTorch can override `CMAKE_CUDA_ARCHITECTURES`. Check the *"Added CUDA NVCC flags"* line in the
configure output to see what was actually used.

### Stale cache after changing toolchains

Use a fresh build directory, or `-Clean` with the scripts, whenever you change the CUDA toolkit,
LibTorch, compiler or `CPU_ONLY`.

## Ubuntu

### Don't install `nvidia-cuda-toolkit`

Ubuntu's `nvidia-cuda-toolkit` package is an old CUDA in non-standard paths. Remove it
(`sudo apt remove nvidia-cuda-toolkit`) and install the toolkit from NVIDIA's repository as shown
in [Installation](../installation/).

### `libtorch_cuda.so: cannot open shared object file`

Running a program needs LibTorch's `lib` folder on the loader path:

```bash
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

Alternatively, set an rpath in CMake (`set(CMAKE_BUILD_RPATH ${TORCH_INSTALL_PREFIX}/lib)`).

## Windows

### The program exits immediately with `0xC0000135`

A DLL is missing. Copy LibTorch's DLLs next to your `.exe` (the [quick start](../quickstart/)
CMake file does this), or add `<libtorch>\lib` to `PATH`. The CUDA runtime DLLs
(`%CUDA_PATH%\bin` and, on CUDA 13, `%CUDA_PATH%\bin\x64`) must also be on `PATH`.

### The program exits with `0xC0000409` and prints nothing

The program threw a `c10::Error` (a failed `TORCH_CHECK`) that nothing caught. Wrap `main` to see
the message:

```cpp
int main() try {
  // ...
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
```

### `OMP: Error #15: Initializing libiomp5md.dll, but found libomp140 already initialized`

Two OpenMP runtimes were loaded. LibTorch ships `libiomp5md`, so don't add `/openmp` or another
OpenMP runtime to your own targets. As a last resort, set `KMP_DUPLICATE_LIB_OK=TRUE`.

### `unsupported Microsoft Visual Studio version`

nvcc rejects MSVC versions newer than it knows. Add
`-DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"`; RiemannEngine's own build does this
automatically.

### `M_PI` is undefined

MSVC hides it by default. Define `_USE_MATH_DEFINES` before including `<cmath>`, or use
`std::numbers::pi` (C++20).

## Runtime

### The program runs on CPU

`torch::cuda::is_available()` returns false. Check, in this order:

1. `nvidia-smi` works and shows your GPU. If it doesn't, install or update the NVIDIA driver.
2. You linked a **CUDA** LibTorch, not the CPU build. `TORCH_INSTALL_PREFIX/lib` should contain
   `torch_cuda.dll` or `libtorch_cuda.so`.
3. On Windows, the linker can drop `torch_cuda.dll` because nothing references it directly. The
   `torch_cuda` CMake target forces it in with `-INCLUDE:?warp_size@cuda@at@@YAHXZ`, so link
   `${TORCH_LIBRARIES}` through CMake rather than listing `torch.lib` by hand.
4. The driver supports the CUDA version of your LibTorch. `nvidia-smi` prints the highest
   supported CUDA version in its header.
5. In Docker, the container was started with `--gpus all` and the NVIDIA Container Toolkit is
   installed. See [Docker troubleshooting](../docker/#troubleshooting).

### `Expected all tensors to be on the same device`

Move the model with `.to(device)` and create both features *and* coordinates on that same device.

### Labels or skip connections don't line up

Quantization and striding can change the number and order of rows. Map labels through
`SparseQuantizeResult::inverse_mapping` or a [`TensorField`](../sparse-tensors/#tensorfield), and
build skip connections from tensors that share a coordinate key (see
[Core concepts](../concepts/)). Two tensors having the same row count does not mean they share a
key.

### Gaussian rendering errors

- Rendering needs **float32** inputs on one device. `GaussianTensor` itself also accepts float64.
- Quaternions are **w, x, y, z**.
- Targets are `[H, W, 3]` float32 images in [0, 1], with calibrated cameras.
- The PLY loader reads binary little-endian files. Use `load_gaussian_model_ply` to keep the
  higher-order SH coefficients.
- Keep models and tensors alive for as long as an optimizer or trainer references them.
- Cap the Gaussian count (`max_gaussians`) in long densification runs.

## Building packages

### Docker build fails

- Docker Desktop must run **Linux containers**.
- Each profile needs about 30 GB free disk space.
- Lower `-Jobs` / `--jobs` if the compiler is killed (out of memory).
- Look at the **first** failing step (download, CMake, compile or packaging).

### NVTX errors with CUDA 13

CUDA 13's versioned NVTX namespaces clash with the legacy NVTX wrapper. The engine uses
`<nvtx3/nvtx3.hpp>` when available. To check your toolchain:

```bash
CHECK_CUDA=1 bash tests/builders/test-builders.sh
```

## Known limitations

| Area | Limitation |
|---|---|
| Adaptive engine | Hierarchy and kernel maps are built on the host. See [Adaptive limitations](../adaptive/#limitations) |
| Uniform resampling | Output can be very large for big scale changes |
| Gaussian CUDA rasterizer | Fixed 16×16 tiles; `tile_size` is ignored on the fused CUDA path |
| Gaussian / sparse voxels | No direct bridge between `GaussianModel` and `SparseTensor` |
| Compressed Gaussian PLY | SH decoding not verified against a reference |
| Platforms | x86-64 only. No ROCm or macOS support |

## Reporting an issue

[Open an issue](https://github.com/QuiteLiterallyConnor/RiemannEngine/issues) with:

- OS, GPU and driver version (`nvidia-smi`);
- the RiemannEngine variant, LibTorch and CUDA versions, compiler and CMake versions;
- the **first** error message and a minimal program that reproduces it.

For performance reports also include problem sizes, dtypes, the device, Release/Debug, warm-up and
how you timed it.
