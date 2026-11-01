---
title: Installation
nav_order: 2
description: Install a prebuilt RiemannEngine SDK on Windows or Ubuntu, or use the Docker image.
permalink: /installation/
---

# Installation
{: .no_toc }

RiemannEngine ships as a prebuilt **static library + headers + CMake package**. Your program
also links against **LibTorch** and the **CUDA toolkit**, and both must match the variant you
install.
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## Choose a variant

Each release is built once per CUDA version. Pick the row that matches your OS and the CUDA version
you have, or want, installed.

| Variant | OS | CUDA toolkit | cuDNN | LibTorch | Compiler |
|---|---|---|---|---|---|
| `cu132` | Windows 10/11 x64 | 13.2 | 9.x for CUDA 13 | 2.15.0+cu132 | VS 2022 (MSVC 14.4x) |
| `cu134` | Windows 10/11 x64 | 13.4 | 9.x for CUDA 13 | 2.15.0+cu134 | VS 2022 (MSVC 14.4x) |
| `cu124` | Ubuntu 22.04 x64 | 12.4 | 9.x for CUDA 12 | 2.5.1+cu124 (CXX11 ABI) | GCC 11 |
| `cu132` | Ubuntu 24.04 x64 | 13.2 | 9.x for CUDA 13 | 2.15.0+cu132 | GCC 13 |
| `cu134` | Ubuntu 24.04 x64 | 13.4 | 9.x for CUDA 13 | 2.15.0+cu134 | GCC 13 |

{: .important }
> **GPU architecture.** Release binaries contain kernels for compute capability **8.6**
> (RTX 30-series / A-series Ampere). Other GPUs need a [source build](../building/)
> with `CMAKE_CUDA_ARCHITECTURES` set to your SM number. CUDA 13 cannot target GPUs older than
> Turing (SM 7.5); use the `cu124` variant or a CUDA 12 source build for those.

## Release assets

Download from the [Releases page](https://github.com/QuiteLiterallyConnor/RiemannEngine/releases).

| Asset | Contents | Installs / extracts to |
|---|---|---|
| `RiemannEngine-<ver>-cu<X>-x64.msi` | Windows installer | `C:\Program Files\RiemannEngine\<ver>_cu<X>\` |
| `RiemannEngine-<ver>-cu<X>-win-x64.zip` | Windows SDK, no installer | `RiemannEngine\<ver>_cu<X>\` |
| `libriemann-cpp-dev_<ver>_cu<X>_ubuntu<U>_amd64.deb` | Ubuntu package | `/usr/local/RiemannEngine/<ver>_cu<X>/` |
| `RiemannEngine-<ver>-cu<X>-ubuntu<U>-x64.tar.gz` | Ubuntu SDK, no package manager | `RiemannEngine/<ver>_cu<X>/` |
| `SHA256SUMS.txt` | Checksums for every asset | |

Every install has the same layout, and variants install **side by side**:

```
RiemannEngine/<ver>_cu<X>/
├── include/riemann/              # #include <riemann.hpp>
│   ├── riemann.hpp               # umbrella header
│   ├── engine/ octree/ splat/    # canonical headers
│   └── layers/ dynamic/ normal/  # forwarding headers
└── lib/
    ├── riemann_cpp.lib           # Windows (libriemann_cpp.a on Linux)
    ├── libiomp5md.lib            # Windows only: OpenMP import lib matching LibTorch
    └── cmake/RiemannEngine/      # RiemannEngineConfig.cmake + targets
```

Verify a download:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing          # Linux
```
```powershell
Get-FileHash .\RiemannEngine-1.0.0-cu134-x64.msi       # Windows, compare by eye
```

## Windows

### 1. Install RiemannEngine

```powershell
msiexec /i RiemannEngine-1.0.0-cu134-x64.msi
```

Or extract the `.zip` anywhere, for example `C:\`, to get `C:\RiemannEngine\1.0.0_cu134`.

### 2. Install the prerequisites

| Requirement | How |
|---|---|
| NVIDIA driver | Latest [GeForce/RTX/Studio driver](https://www.nvidia.com/Download/index.aspx). `nvidia-smi` must report a CUDA version ≥ your variant. |
| CUDA Toolkit 13.2 / 13.4 | [CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive). The installer sets `CUDA_PATH`. |
| cuDNN 9 | [cuDNN downloads](https://developer.nvidia.com/cudnn-downloads) → *Windows / x86_64 / CUDA 13* |
| Visual Studio 2022 | *Desktop development with C++* workload |
| CMake ≥ 3.24, Ninja | `winget install Kitware.CMake Ninja-build.Ninja` |
| LibTorch 2.15.0 | See below |

```powershell
$cu = "cu134"   # or cu132
curl.exe -L -o libtorch.zip "https://download.pytorch.org/libtorch/$cu/libtorch-win-shared-with-deps-2.15.0%2B$cu.zip"
tar -xf libtorch.zip -C C:\
Rename-Item C:\libtorch "C:\libtorch_$cu"
```

{: .warning }
> Use the **Release** LibTorch build. The Debug build is ABI-incompatible with the Release
> `riemann_cpp.lib`.

### 3. Runtime DLLs

Programs need LibTorch's DLLs (`torch_cuda.dll`, `c10.dll`, and others) at run time. Either add
`C:\libtorch_cu134\lib` to `PATH`, or copy the DLLs next to your `.exe`. The
[quick-start CMake file](../quickstart/#option-a-cmake) does the copying for you.

## Ubuntu

### 1. Install RiemannEngine

```bash
sudo apt install ./libriemann-cpp-dev_1.0.0_cu134_ubuntu24.04_amd64.deb
```

Or, without a package manager:

```bash
sudo tar -xzf RiemannEngine-1.0.0-cu134-ubuntu24.04-x64.tar.gz -C /usr/local
```

Uninstall with `sudo apt remove libriemann-cpp-dev-1.0.0-cu134`.

### 2. Install the prerequisites

```bash
# NVIDIA driver
sudo ubuntu-drivers install && sudo reboot

# CUDA + cuDNN from NVIDIA's repository (use ubuntu2204 on 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-4 cudnn9-cuda-13     # cu134
# sudo apt install cuda-toolkit-13-2 cudnn9-cuda-13   # cu132
# sudo apt install cuda-toolkit-12-4 cudnn9-cuda-12   # cu124 (22.04)

# Build tools
sudo apt install build-essential cmake ninja-build unzip wget libgomp1
```

{: .note }
> Don't install Ubuntu's own `nvidia-cuda-toolkit` package. It is CUDA 12.0 and conflicts with
> NVIDIA's packages.

### 3. Install LibTorch

```bash
cu=cu134   # or cu132
wget -O libtorch.zip "https://download.pytorch.org/libtorch/$cu/libtorch-shared-with-deps-2.15.0%2B$cu.zip"
# cu124 (Ubuntu 22.04):
# wget -O libtorch.zip "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip"
sudo unzip -q libtorch.zip -d /opt          # -> /opt/libtorch
```

Pass the LibTorch folder to CMake with `-DCMAKE_PREFIX_PATH=/opt/libtorch`; your own
`find_package(Torch)` call needs it. (RiemannEngine's package also falls back to `/opt/libtorch` when
it looks for Torch itself.)

## Docker

The [`acaicia/riemannengine`](https://hub.docker.com/r/acaicia/riemannengine) images contain
CUDA, cuDNN, LibTorch, CMake, Ninja and RiemannEngine, already built and installed to `/usr/local`.

| Tag | Ubuntu | CUDA | LibTorch |
|---|---|---|---|
| `cu134`, `latest`, `<ver>-cu134` | 24.04 | 13.4.3 | 2.15.0 |
| `cu132`, `<ver>-cu132` | 24.04 | 13.2.1 | 2.15.0 |
| `cu124`, `<ver>-cu124` | 22.04 | 12.4.0 | 2.5.1 |

Host requirements:

- **Linux:** Docker Engine plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- **Windows:** Docker Desktop with the WSL 2 backend and a current NVIDIA driver.

```bash
docker run --gpus all --rm -it -v "$(pwd):/workspace" acaicia/riemannengine:cu134
# inside the container:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch
cmake --build build
```

| Path in the image | Contents |
|---|---|
| `/usr/local/include/riemann` | Headers |
| `/usr/local/lib/libriemann_cpp.a` | Static library |
| `/usr/local/lib/cmake/RiemannEngine` | CMake package |
| `/opt/libtorch` | LibTorch |
| `/workspace` | Working directory |

See the **[Docker guide](../docker/)** for host setup, file ownership, Compose, VS Code Dev
Containers, slim deployment images, supported GPUs and troubleshooting.

## Verify the install

Build and run the [quick-start program](../quickstart/). It prints the version and the device,
and runs a small network on the GPU.
