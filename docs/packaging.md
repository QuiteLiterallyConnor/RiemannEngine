---
title: Packaging & releases
nav_order: 13
description: How the release artifacts (MSI, .deb, SDK archives, Docker images) are produced and published.
permalink: /packaging/
---

# Packaging & releases
{: .no_toc }

This page is for maintainers. One script produces every artifact on the
[releases page](https://github.com/QuiteLiterallyConnor/RiemannEngine/releases).
{: .fs-5 .fw-300 }

1. TOC
{:toc}

---

## One-command release

```powershell
.\scripts\build-release.ps1                    # everything; pushes Docker images
.\scripts\build-release.ps1 -NoPush -Clean     # clean Windows builds, keep images local
.\scripts\build-release.ps1 -SkipDocker -WindowsCuda 134 -DebCuda 134
```

The version comes from `project(VERSION)` in `CMakeLists.txt`. Artifacts go to `dist\<version>\`.
A failed variant does not stop the others. A summary is printed at the end, and the script exits
non-zero if anything failed (`-StopOnError` aborts at the first failure instead).

### Artifacts

| Artifact | Variants | Installs / extracts to |
|---|---|---|
| `RiemannEngine-<ver>-cu<X>-x64.msi` | cu132, cu134 | `<ProgramFiles>\RiemannEngine\<ver>_cu<X>\` |
| `RiemannEngine-<ver>-cu<X>-win-x64.zip` | cu132, cu134 | `RiemannEngine\<ver>_cu<X>\{include,lib}` |
| `libriemann-cpp-dev_<ver>_cu<X>_ubuntu<U>_amd64.deb` | 22.04/cu124, 24.04/cu132, 24.04/cu134 | `/usr/local/RiemannEngine/<ver>_cu<X>/` |
| `RiemannEngine-<ver>-cu<X>-ubuntu<U>-x64.tar.gz` | same as the .deb | `RiemannEngine/<ver>_cu<X>/{include,lib}` |
| Docker `acaicia/riemannengine:<ver>-cu<X>` and `:cu<X>` | cu124, cu132, cu134 | `:latest` points at cu134 |
| `SHA256SUMS.txt` | | Checksums of every file above |

Archives contain only RiemannEngine: headers, the static library and the CMake package. Users
supply LibTorch and CUDA themselves (see [Installation](../installation/)). Every version/CUDA
pair installs to its own directory, so they co-exist.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `-Version` | from `CMakeLists.txt` | Release version |
| `-WindowsCuda` | `132,134` | Windows variants (LibTorch from `D:\libtorch_cu<X>`) |
| `-DebCuda` | `124,132,134` | Ubuntu .deb and tarball variants |
| `-DockerCuda` | `124,132,134` | Docker images |
| `-OutputDir` | `dist\<version>` | Artifact directory |
| `-Jobs` | `12` | Parallel compile jobs |
| `-CudaArchitectures` | `86` | SM list for the Linux builds, e.g. `"86;89;90"` |
| `-DockerUser` / `-DockerRepo` | `$env:DOCKER_USER` or `acaicia` / `riemannengine` | Image name |
| `-SkipWindows`, `-SkipDeb`, `-SkipDocker` | | Skip a stage |
| `-NoPush` | | Build images without pushing |
| `-Clean` | | Wipe the `build-release-cu<X>` directories first |
| `-StopOnError` | | Abort at the first failure |

### Requirements

- Everything needed for a [Windows build](../building/#windows): VS 2022, CUDA 13.2 and 13.4,
  `D:\libtorch_cu132` and `D:\libtorch_cu134`.
- WiX Toolset v4+ and its UI extension (below).
- Docker Desktop in Linux containers mode, with roughly 30 GB free per Linux profile.
- `docker login` when pushing.

## Windows MSI

```powershell
dotnet tool install --global wix
wix extension add -g WixToolset.UI.wixext/<version shown by `wix --version`>

.\msi\build-msi.ps1 -BuildDir .\build-cu134 -CudaTag 134
```

`build-msi.ps1` runs `cmake --install` into a staging directory and builds the MSI from it.
Parameters: `-Version`, `-CudaTag` (detected from `CMakeCache.txt`), `-BuildDir`, `-BuildType`,
`-StagingDir`, `-OutputPath`, `-SkipStaging`, `-CpuOnly`.

The MSI:

- installs to `C:\Program Files\RiemannEngine\<version>_<variant>\`;
- registers the prefix in the CMake package registry
  (`HKLM\Software\Kitware\CMake\Packages\RiemannEngine`), so `find_package` works without
  `CMAKE_PREFIX_PATH`;
- uses a separate UpgradeCode for every version/variant pair, so cu132 and cu134, and different
  versions, all install side by side. Reinstalling the same version and variant replaces it.

## Ubuntu .deb and tarball

```powershell
.\apt\build-apt.ps1 -CudaVersion 134            # Windows host, via Docker
```

```bash
./apt/build-apt.sh --cuda-version 134           # Linux host with CUDA + LibTorch installed
```

| Profile | CUDA | Ubuntu | LibTorch |
|---|---|---|---|
| `134` (default) | 13.4.3 + cuDNN | 24.04 | 2.15.0 / cu134 |
| `132` | 13.2.1 + cuDNN | 24.04 | 2.15.0 / cu132 |
| `124` (legacy) | 12.4.0 | 22.04 | 2.5.1 / cu124 |

The package is named `libriemann-cpp-dev-<ver>-cu<X>`, so different versions and profiles can be
installed together. It does **not** depend on Ubuntu's `nvidia-cuda-toolkit`. Users install CUDA
from NVIDIA's repository.

## Docker images

```powershell
.\docker\docker-build.ps1 -CudaVersion 134 -Push         # acaicia/riemannengine:<ver>-cu134, :cu134
```

```bash
./docker/docker-build.sh --cuda-version 134 --push
```

The image name comes from `-User`/`-Repo` (`--user`/`--repo`), or the `DOCKER_USER` and
`DOCKER_REPO` environment variables. `build-release.ps1` also tags cu134 as `latest`.

## Publishing a release

1. Bump `project(VERSION ...)` in `CMakeLists.txt`.
2. Run `.\scripts\build-release.ps1 -Clean`.
3. Verify: `Get-FileHash` against `SHA256SUMS.txt`; install one MSI and one .deb and build the
   [quick start](../quickstart/) program against each.
4. Create a GitHub release tagged `v<version>` and upload everything in `dist\<version>\`.

## Documentation site

This site is built by `.github/workflows/pages.yml` when a push to `main` touches `docs/`, the
public headers or the workflow. Pull requests run the same build without deploying.

1. Jekyll (just-the-docs theme) renders `docs/`.
2. Doxygen (`docs/Doxyfile`) generates the API reference into `_site/api/reference/`.
3. The result is deployed with `actions/deploy-pages`.

In the repository settings, set **Pages > Source** to **GitHub Actions**. To preview locally (Ruby
3.x):

```bash
cd docs
bundle install
bundle exec jekyll serve          # http://localhost:4000/RiemannEngine/
cd ..
doxygen docs/Doxyfile              # from the repo root -> build/doxygen/html/index.html
```
