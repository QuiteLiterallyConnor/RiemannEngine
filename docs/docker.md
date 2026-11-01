---
title: Docker
nav_order: 4
description: Develop, build, run and deploy RiemannEngine programs with the GPU-enabled Docker images.
permalink: /docker/
---

# Docker
{: .no_toc }

The [`acaicia/riemannengine`](https://hub.docker.com/r/acaicia/riemannengine) images are complete
GPU development environments: CUDA, LibTorch, CMake, Ninja, GCC and a prebuilt RiemannEngine. You
only need Docker and an NVIDIA driver on the host.
{: .fs-5 .fw-300 }

[Docker Hub](https://hub.docker.com/r/acaicia/riemannengine){: .btn .btn-primary }
[Dockerfile](https://github.com/QuiteLiterallyConnor/RiemannEngine/blob/main/docker/Dockerfile){: .btn }

1. TOC
{:toc}

---

## What's in the image

### Tags

| Tag | Ubuntu | CUDA | cuDNN | LibTorch | GCC | CMake |
|---|---|---|---|---|---|---|
| `cu134`, `latest`, `<ver>-cu134` | 24.04 | 13.4.3 | 9 | 2.15.0 | 13 | 4.2.3 |
| `cu132`, `<ver>-cu132` | 24.04 | 13.2.1 | 9 | 2.15.0 | 13 | 4.2.3 |
| `cu124`, `<ver>-cu124` | 22.04 | 12.4.0 | bundled with LibTorch | 2.5.1 | 11 | 3.28.6 |

Use `<ver>-cu<X>` (e.g. `1.0.0-cu134`) to pin a release, and `cu<X>` to follow the newest one.
The images are about 21 GB because they include the full CUDA toolkit and LibTorch.

### Layout

| Path | Contents |
|---|---|
| `/usr/local/include/riemann/` | RiemannEngine headers (`riemann.hpp`, `engine/`, `octree/`, `splat/`, ...) |
| `/usr/local/lib/libriemann_cpp.a` | RiemannEngine static library |
| `/usr/local/lib/cmake/RiemannEngine/` | CMake package (`RiemannEngine::RiemannEngine`) |
| `/opt/libtorch/` | LibTorch (headers, shared libraries, CMake package) |
| `/usr/local/cuda/` | CUDA toolkit (`nvcc`, headers, libraries) |
| `/workspace` | Default working directory. Mount your project here |

### Environment

| Variable | Value |
|---|---|
| `CUDA_HOME` | `/usr/local/cuda` |
| `PATH` | Includes `/usr/local/cuda/bin` |
| `LD_LIBRARY_PATH` | Includes `/usr/local/cuda/lib64` and `/opt/libtorch/lib`, so built programs run without extra setup |

The default command is `bash`, and the default user is `root` (see [file ownership](#file-ownership-on-linux)).

### Supported GPUs

{: .important }
> The published images are compiled for **SM 8.6** only. That covers the RTX 30 and RTX 40 series,
> A10, A40, L4 and L40 (compute capability 8.6 and 8.9). On other GPUs, such as T4 (7.5),
> A100 (8.0), H100 (9.0) or RTX 50 (12.0), RiemannEngine's kernels fail with *"no kernel image is
> available for execution on the device"*. [Build the image yourself](#building-the-image-yourself)
> with your architecture in that case.

Check your GPU's compute capability with:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Setting up the host

You need a recent NVIDIA driver (`nvidia-smi` must work on the host), Docker, and GPU support for
Docker. The driver must support the image's CUDA version: **580+** for the CUDA 13 tags, **550+**
for `cu124`.

### Linux

1. Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. Register it with Docker and restart:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. Optional: run Docker without `sudo` with `sudo usermod -aG docker $USER`, then log out and back
   in.

### Windows

1. Install a current NVIDIA Game Ready or Studio driver. WSL 2 GPU support comes with it, so don't
   install a separate driver inside WSL.
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and keep the default
   **WSL 2 backend** and **Linux containers** mode.

GPU support works without any further setup.

{: .tip }
> On Windows, keep projects you build often inside the WSL file system (e.g. `\\wsl$\Ubuntu\home\you\project`)
> rather than on `C:\`. Bind mounts from Windows drives work but file access through them is much
> slower, which makes builds slower.

### Verify

```bash
docker run --rm --gpus all acaicia/riemannengine:cu134 nvidia-smi
```

This should print the same GPU table as `nvidia-smi` on the host. If it fails, see
[Troubleshooting](#troubleshooting).

## Running the container

### Interactive shell

From your project folder:

Linux, macOS or WSL shell
{: .label }

```bash
docker run --gpus all --rm -it -v "$(pwd):/workspace" acaicia/riemannengine:cu134
```

PowerShell
{: .label }

```powershell
docker run --gpus all --rm -it -v "${PWD}:/workspace" acaicia/riemannengine:cu134
```

Command Prompt
{: .label }

```bat
docker run --gpus all --rm -it -v "%cd%:/workspace" acaicia/riemannengine:cu134
```

| Flag | Meaning |
|---|---|
| `--gpus all` | Give the container every host GPU. **Without it, programs silently fall back to CPU** |
| `--rm` | Delete the container on exit. Files under mounted folders stay on the host |
| `-it` | Interactive terminal |
| `-v <host>:/workspace` | Mount your project at `/workspace` (the working directory) |

Everything outside mounted folders is lost when a `--rm` container exits.

### Running a single command

Pass a command after the image name to run it and exit:

```bash
docker run --gpus all --rm -v "$(pwd):/workspace" acaicia/riemannengine:cu134 \
  bash -c "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch \
           && cmake --build build && ./build/hello_riemann"
```

This suits CI jobs and scripts.

### A long-lived container

To keep installed packages and shell history between sessions, create a named container once and
reuse it:

```bash
docker run --gpus all -dit --name riemann-dev -v "$(pwd):/workspace" acaicia/riemannengine:cu134
docker exec -it riemann-dev bash        # open a shell (repeat as often as you like)
docker stop riemann-dev                  # later: docker start riemann-dev
docker rm -f riemann-dev                 # delete it
```

## Building your program

Use the program from the [Quick start](../quickstart/) (`main.cpp`) or your own. All three build
methods below were tested inside the `cu134` image.

### CMake

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
```

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch
cmake --build build
./build/hello_riemann
```

```
RiemannEngine 1.0.0 on cuda
input voxels : 761
logits shape : [2, 4]
...
[ CUDAFloatType{2,4} ]
```

{: .note }
> `-DCMAKE_PREFIX_PATH=/opt/libtorch` is required so that `find_package(Torch)` finds LibTorch.
> RiemannEngine itself is in `/usr/local`, which CMake searches by default.

The same `CMakeLists.txt` works unchanged with the Windows and Ubuntu installs.

### Makefile or g++

Use the [Makefile from the Quick start](../quickstart/) with RiemannEngine's prefix set to `/usr/local`:

```bash
make RIEMANN=/usr/local
```

Or call `g++` directly:

```bash
g++ -std=c++20 -O2 -fopenmp main.cpp -o hello_riemann \
    -DRIEMANN_NO_PYBIND -DUSE_CUDA -DRIEMANN_ENABLE_GAUSSIAN \
    -I/usr/local/include/riemann -I/usr/local/include/riemann/vendor \
    -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include \
    -I/usr/local/cuda/include \
    -L/usr/local/lib -L/opt/libtorch/lib -L/usr/local/cuda/lib64 \
    -lriemann_cpp \
    -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed \
    -ltorch -ltorch_cpu -lc10 -lcudart -lcublas -lcusparse \
    -Wl,-rpath,/opt/libtorch/lib -Wl,-rpath,/usr/local/cuda/lib64
```

`--no-as-needed` keeps `libtorch_cuda` linked. Without it, LibTorch never registers its CUDA
backend and everything runs on the CPU.

### Writing your own CUDA kernels

`nvcc` is available, so you can add `.cu` files. Enable CUDA in your project and set your GPU's
architecture:

```cmake
project(my_app LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)
add_executable(my_app main.cpp my_kernels.cu)
target_link_libraries(my_app PRIVATE RiemannEngine::RiemannEngine)
```

## Working with data

### File ownership on Linux

The container runs as `root`, so on a Linux host the files it creates in a mounted folder (such as
`build/`) belong to root. Run as your own user instead:

```bash
docker run --gpus all --rm -it --user "$(id -u):$(id -g)" \
  -v "$(pwd):/workspace" acaicia/riemannengine:cu134
```

Building and running work fine as a non-root user. Your shell prompt may show `I have no name!`
if your UID doesn't exist in the image; this is harmless. (UID 1000 maps to the image's `ubuntu`
user.) Docker Desktop on Windows handles ownership for you.

### Datasets

Mount datasets separately, read-only, so a bug can't modify them:

```bash
docker run --gpus all --rm -it \
  -v "$(pwd):/workspace" \
  -v /mnt/datasets/scannet:/data:ro \
  acaicia/riemannengine:cu134
```

Write checkpoints and outputs to `/workspace` (or another mounted folder) so they survive the
container.

### Choosing GPUs

```bash
docker run --gpus all ...                 # every GPU
docker run --gpus 1 ...                   # any one GPU
docker run --gpus '"device=0"' ...        # GPU 0
docker run --gpus '"device=1,2"' ...      # GPUs 1 and 2
```

Inside the container, the visible GPUs are renumbered from 0. Use `torch::Device(torch::kCUDA, i)`
to pick one in code.

## Docker Compose

A `compose.yaml` in your project saves typing the flags:

```yaml
services:
  dev:
    image: acaicia/riemannengine:cu134
    working_dir: /workspace
    volumes:
      - .:/workspace
      - ./data:/data:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all            # or: device_ids: ["0"]
              capabilities: [gpu]
    stdin_open: true
    tty: true
```

```bash
docker compose run --rm dev                      # interactive shell
docker compose run --rm dev bash -c "cmake --build build && ./build/train"
```

## VS Code Dev Containers

With the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension, VS Code can open your project inside the image, so IntelliSense, building and debugging
all use the container's toolchain. Add `.devcontainer/devcontainer.json`:

```json
{
  "name": "RiemannEngine",
  "image": "acaicia/riemannengine:cu134",
  "runArgs": ["--gpus", "all"],
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
  "customizations": {
    "vscode": {
      "extensions": ["ms-vscode.cpptools", "ms-vscode.cmake-tools"],
      "settings": {
        "cmake.generator": "Ninja",
        "cmake.configureSettings": { "CMAKE_PREFIX_PATH": "/opt/libtorch" }
      }
    }
  }
}
```

Then run **Dev Containers: Reopen in Container** from the command palette.

## Deploying your program

The development image is large because it contains compilers and the CUDA toolkit. A program only
needs its own binary plus LibTorch's `lib/` folder, which already bundles the CUDA runtime, cuBLAS,
cuSPARSE, cuDNN and OpenMP. A two-stage Dockerfile builds in the dev image and ships on plain
Ubuntu:

```dockerfile
# ---- build stage: the RiemannEngine development image ----
FROM acaicia/riemannengine:cu134 AS build
WORKDIR /src
COPY CMakeLists.txt main.cpp ./
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch \
 && cmake --build build

# ---- runtime stage: no compiler, no CUDA toolkit ----
FROM ubuntu:24.04
COPY --from=build /opt/libtorch/lib /opt/libtorch/lib
COPY --from=build /src/build/hello_riemann /usr/local/bin/hello_riemann
ENV LD_LIBRARY_PATH=/opt/libtorch/lib
ENTRYPOINT ["hello_riemann"]
```

```bash
docker build -t hello-riemann .
docker run --rm --gpus all hello-riemann
```

The result is about 5 GB instead of 21 GB, and runs on any host that can run the dev image. The
NVIDIA Container Toolkit injects the driver at run time, so the runtime stage needs no CUDA base
image. Add a `.dockerignore` containing `build/` so local build folders aren't sent to Docker.

{: .tip }
> Base the runtime stage on the same Ubuntu release as the build stage (`ubuntu:22.04` for `cu124`),
> so the C++ runtime library matches.

## Building the image yourself

Build your own image to target other GPU architectures, or to include local changes to
RiemannEngine. From a clone of the repository:

PowerShell
{: .label }

```powershell
.\docker\docker-build.ps1 -CudaVersion 134 -CudaArchitectures "75;80;86;89;90;120" -Jobs 8 -Tag my-cu134
```

bash
{: .label }

```bash
./docker/docker-build.sh --cuda-version 134 --cuda-architectures "75;80;86;89;90;120" --jobs 8 --tag my-cu134
```

Or with `docker build` directly:

```bash
docker build -f docker/Dockerfile -t riemannengine:my-cu134 \
  --build-arg CUDA_VERSION=134 \
  --build-arg CUDA_ARCHITECTURES="86;89;120" \
  --build-arg JOBS=8 .
```

| Build argument | Default | Meaning |
|---|---|---|
| `CUDA_VERSION` | `134` | `124`, `132` or `134` (selects the base image and LibTorch) |
| `CUDA_ARCHITECTURES` | `86` | Semicolon-separated SM numbers. CUDA 13 needs 75 or newer |
| `JOBS` | `4` | Parallel compile jobs. Lower it if the build runs out of memory |

Each extra architecture lengthens the build. Allow about 30 GB of free disk space per CUDA
version. Publishing images is covered in [Packaging](../packaging/#docker-images).

## Troubleshooting

### `could not select device driver "" with capabilities: [[gpu]]`

Docker can't find the NVIDIA runtime. On Linux, install the NVIDIA Container Toolkit and run
`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. On Windows,
update Docker Desktop and the NVIDIA driver, and make sure the WSL 2 backend is enabled.

### The program prints `on cpu`

The container was started without `--gpus all`. The CUDA banner at startup then says *"The NVIDIA
Driver was not detected"*. Also check `nvidia-smi` inside the container.

### `CUDA driver version is insufficient for CUDA runtime version`

The host driver is older than the image's CUDA. Update the driver, or use an older tag (`cu124`
needs a 550+ driver).

### `no kernel image is available for execution on the device`

Your GPU's architecture isn't compiled into the image (see [Supported GPUs](#supported-gpus)).
[Build the image](#building-the-image-yourself) with `CUDA_ARCHITECTURES` set to your compute
capability (e.g. `90` for H100).

### `Could not find a package configuration file provided by "Torch"`

Add `-DCMAKE_PREFIX_PATH=/opt/libtorch` to the CMake configure command.

### Permission denied on `build/` after using the container

The files were created by root. Remove them with
`docker run --rm -v "$(pwd):/workspace" acaicia/riemannengine:cu134 rm -rf build` and use
[`--user`](#file-ownership-on-linux) from now on.

### Builds are slow on Windows

Move the project into the WSL file system (see the tip in [Windows](#windows)), and give Docker
Desktop more CPUs and memory under **Settings > Resources**.

### `no space left on device`

The images are large. Remove unused images and build cache with `docker image prune` and
`docker builder prune`. On Windows, Docker Desktop's disk limit is under **Settings > Resources**.

### `/bin/bash^M: bad interpreter`

A script was saved with Windows line endings. Convert it with `sed -i 's/\r$//' script.sh`, or set
your editor or Git (`core.autocrlf=input`) to use LF for shell scripts.
