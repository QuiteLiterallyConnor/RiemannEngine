# apt — Debian Package Builder

Produces `libriemann-cpp-dev_<VERSION>_cu<PROFILE>_ubuntu<UBUNTU>_amd64.deb` and a
matching SDK tarball `RiemannEngine-<VERSION>-cu<PROFILE>-ubuntu<UBUNTU>-x64.tar.gz`.
The Docker path defaults to **CUDA 13.4**, with selectable modern and legacy profiles:

| Profile | CUDA toolkit | Ubuntu | LibTorch (CXX11 ABI) |
|---|---|---|---|
| `134` (default) | 13.4.3 + cuDNN | 24.04 | 2.15.0 / cu134 |
| `132` | 13.2.1 + cuDNN | 24.04 | 2.15.0 / cu132 |
| `124` (legacy) | 12.4.0 | 22.04 | 2.5.1 / cu124 |

Each package is named `libriemann-cpp-dev-<VERSION>-cu<PROFILE>` and installs to
`/usr/local/RiemannEngine/<VERSION>_cu<PROFILE>/`, so different versions and CUDA
profiles can be co-installed.
Two build paths are provided:

| Script | Host | Requires |
|---|---|---|
| `build-apt.ps1` | Windows | Docker Desktop (Linux containers mode) |
| `build-apt.sh` | Linux | CMake, Ninja, LibTorch, CUDA toolkit |

---

## Prerequisites

### Windows (`build-apt.ps1`)

- Docker Desktop with Linux containers mode enabled (BuildKit on by default since v23)
- At least 30 GB free disk space per profile for base images, LibTorch and builds

### Linux (`build-apt.sh`)

```bash
sudo apt install build-essential cmake ninja-build dpkg-dev libomp-dev
```

- CUDA Toolkit 12.4, 13.2 or 13.4 (`nvcc` on `PATH`) — detected automatically;
  falls back to CPU-only if absent and no `--cuda-version` was requested
- Matching Linux LibTorch from the table above, extracted (e.g. `/opt/libtorch`)
- CMake 3.24+; 4.2.3 is used by Docker for CUDA 13 support

---

## Quick Start

### Windows — via Docker

```powershell
.\apt\build-apt.ps1
.\apt\build-apt.ps1 -CudaVersion 132
.\apt\build-apt.ps1 -CudaVersion 124
```

### Linux — native (no Docker)

```bash
chmod +x apt/build-apt.sh
./apt/build-apt.sh --libtorch /opt/libtorch
./apt/build-apt.sh --libtorch /opt/libtorch --cuda-version 134
```

The finished package is written to the `apt/` directory in both cases.

---

## Parameters

### `build-apt.ps1`

| Parameter | Default | Description |
|---|---|---|
| `-Version` | from `CMakeLists.txt` | Semver string embedded in the package filename and metadata |
| `-CudaVersion` | `134` | Matched profile: `124`, `132` or `134` |
| `-CudaArchitectures` | `86` | CMake GPU architectures, e.g. `"86;89;90"` |
| `-Jobs` | `4` | Parallel ninja compile jobs inside the container |
| `-OutputDir` | `apt\` | Directory to write the .deb into |
| `-ImageTag` | `riemannengine-deb-builder:cu<PROFILE>` | Docker image tag (used for layer caching) |
| `-KeepImage` | _(switch)_ | Keep the intermediate Docker image after the build |

**Example:**

```powershell
.\apt\build-apt.ps1 -Version 1.2.0 -Jobs 8 -OutputDir .\dist
```

---

### `build-apt.sh`

| Flag | Default | Description |
|---|---|---|
| `--libtorch PATH` | _(required)_ | Path to LibTorch root directory |
| `--version STRING` | `1.0.0` | Semver string embedded in the package |
| `--jobs N` | `nproc` | Parallel ninja compile jobs |
| `--cuda-version N` | detected | Require `124`, `132` or `134`; error if nvcc is missing or mismatched |
| `--cuda-architectures LIST` | `86` | CMake GPU architectures, e.g. `'86;89;90'` |
| `--build-dir PATH` | `<project>/build-deb-cu<PROFILE>` | CMake build directory (`build-deb-cpu` for CPU-only) |
| `--output PATH` | `apt/` | Directory to write the .deb into |
| `--keep-build` | _(flag)_ | Keep the CMake build tree after packaging |

**Example:**

```bash
./apt/build-apt.sh --libtorch /opt/libtorch --version 1.2.0 --jobs 8 --output ./dist
```

---

## What the Build Does

### Windows (Docker path)

1. **Dockerfile stage `builder`** — selects the matched NVIDIA base image and LibTorch download, installs build tools, copies the project source, and runs `cmake --install` into a staging tree. CUDA 13 uses cuDNN development images and CMake 4.2.3.
2. **Package assembly** — writes `DEBIAN/control` and `DEBIAN/postinst`, then calls `dpkg-deb --build`.
3. **Dockerfile stage `artifact`** — `FROM scratch` layer that holds only the .deb, enabling `--output type=local` to stream it directly to the host without running a container.

### Linux (native path)

1. **Configure & build** — runs `cmake` + `ninja` against the project source using the provided LibTorch path.
2. **Stage** — `cmake --install` into a temp directory, then flattens `libriemann_cpp.a` with `ld -r`.
3. **Package** — writes `DEBIAN/control` and `DEBIAN/postinst`, then calls `dpkg-deb --build`.
4. **Cleanup** — removes the build tree (unless `--keep-build` is passed) and the temp staging directory.

---

## What the Package Installs

With `<prefix>` = `/usr/local/RiemannEngine/<VERSION>_cu<PROFILE>` (e.g. `/usr/local/RiemannEngine/1.0.0_cu134`):

| Path | Contents |
|---|---|
| `<prefix>/lib/libriemann_cpp.a` | Static library |
| `<prefix>/include/riemann/` | Public C++ and CUDA headers |
| `<prefix>/lib/cmake/RiemannEngine/` | CMake package config (`find_package` support) |

The SDK tarball contains the same tree under `RiemannEngine/<VERSION>_cu<PROFILE>/`.

---

## Installing on a Target Machine

```bash
sudo dpkg -i libriemann-cpp-dev_1.0.0_cu134_ubuntu24.04_amd64.deb
cmake -DCMAKE_PREFIX_PATH="/usr/local/RiemannEngine/1.0.0_cu134;/path/to/libtorch" ..
```

Install on the matching Ubuntu release with the matching CUDA toolkit and
LibTorch build. Native builds name the file after the host's Ubuntu release
(omitted on other distributions), and CPU-only native builds use the `cpu`
variant instead of `cu<PROFILE>`.
The container path is amd64 only. CUDA 13 cannot compile GPU architectures
below SM 75; use CUDA 12.4 for older GPUs.

**Dependencies** — the package declares `libgomp1` and `libstdc++6` as runtime dependencies (normally already present). LibTorch and CUDA are **not** bundled; install them separately and add LibTorch to `CMAKE_PREFIX_PATH` as shown above.

---

## Files in This Directory

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage Docker build: compile → package → extract artifact |
| `build-apt.ps1` | Windows PowerShell script that drives the Docker build |
| `build-apt.sh` | Linux shell script for a native (no Docker) build |
| `README.md` | This file |
