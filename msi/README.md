# RiemannEngine C++ — MSI Installer

This folder contains tooling to produce a Windows Installer (`.msi`) package for
RiemannEngine C++.

## Prerequisites

| Tool | Install |
|------|---------|
| **.NET SDK 6+** | <https://dotnet.microsoft.com/download> |
| **WiX Toolset v4** | `dotnet tool install --global wix` |
| **WiX UI extension** | `wix extension add -g WixToolset.UI.wixext/<wix version>` (must match `wix --version`) |
| **CMake 3.18+** | <https://cmake.org/download/> |

> You must have already **built** RiemannEngine with CMake before generating
> the MSI.  See [INSTALL.md](../INSTALL.md) for build instructions.

## Quick Start

```powershell
# 1. Build the project (if not already done)
cd <repo-root>
cmake -B build -DCMAKE_PREFIX_PATH="E:\Code\libtorch" -G Ninja
cmake --build build --config Release

# 2. Generate the MSI
cd msi
.\build-msi.ps1 -BuildDir ..\build -CudaTag 134
```

The MSI will be created in this folder as
`RiemannEngine-<version>-<cuXXX|cpu>-x64.msi`. To build every release variant
at once, use [scripts/build-release.ps1](../scripts/build-release.ps1).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Version` | from `CMakeLists.txt` | Semantic version (Major.Minor.Patch) embedded in the MSI. |
| `-CudaTag` | detected from `CMakeCache.txt` | CUDA toolkit the build used, e.g. `132`, `134`. |
| `-BuildDir` | `..\build` | Path to the CMake build directory. |
| `-BuildType` | `Release` | CMake configuration used during the build. |
| `-StagingDir` | `_staging` (inside `msi/`) | Temporary directory for `cmake --install`. |
| `-OutputPath` | auto | Output `.msi` path.  Defaults to `RiemannEngine-<ver>-<variant>-x64.msi`. |
| `-SkipStaging` | off | Reuse an existing staging directory instead of re-running `cmake --install`. |
| `-CpuOnly` | off | Tag the MSI as a CPU-only build (variant `cpu`). |

## What the MSI Installs

The installer places files under
`C:\Program Files\RiemannEngine\<version>_<variant>\` (for example
`1.0.0_cu134`) with the following layout:

```
RiemannEngine\
  1.0.0_cu134\
    lib\
      riemann_cpp.lib          # static library
      cmake\
        RiemannEngine\
          RiemannEngineConfig.cmake
          RiemannEngineConfigVersion.cmake
          RiemannEngineTargets.cmake
          RiemannEngineTargets-release.cmake
    include\
      riemann\
        *.hpp / *.h / *.cuh      # public headers (preserving subdirectory structure)
```

It also registers the install in CMake's system package registry
(`HKLM\Software\Kitware\CMake\Packages\RiemannEngine`, value `<version>_<variant>`).
When several variants are installed, select one explicitly:

```cmake
# cmake -DCMAKE_PREFIX_PATH="C:/Program Files/RiemannEngine/1.0.0_cu134;C:/libtorch" ..
find_package(RiemannEngine CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE RiemannEngine::riemann_cpp)
```

## Upgrade Behaviour

Each version/variant combination gets its own `UpgradeCode` (derived from
`<version>_<variant>`), so different versions and CUDA variants install side by
side and uninstall independently.  Re-installing an MSI rebuilt for the same
version/variant replaces the previous build.

## Troubleshooting

- **"wix not found"** — Run `dotnet tool install --global wix` and make sure
  `~/.dotnet/tools` is on your `PATH`.
- **"WixToolset.UI.wixext" errors** — Run `wix extension add WixToolset.UI.wixext`.
- **Empty staging directory** — Make sure the CMake build completed successfully
  and that `cmake --install` produces output to the staging path.
- **Iterate on WiX source without rebuilding** — Use `-SkipStaging` to skip the
  cmake install step and reuse the existing staging directory.
