# Windows Build Scripts

These PowerShell scripts help build RiemannEngine C++ on Windows with MSVC and CUDA.

## Release builds (`build-release.ps1`)

One command produces every release artifact in `dist\<version>\`. The version
defaults to `project(VERSION)` in `CMakeLists.txt`.

```powershell
.\scripts\build-release.ps1                    # everything, pushes Docker images
.\scripts\build-release.ps1 -NoPush -Clean     # clean Windows builds, keep images local
.\scripts\build-release.ps1 -SkipDocker -WindowsCuda 134 -DebCuda 134
```

| Artifact | Variants | Installs / extracts to |
|---|---|---|
| `RiemannEngine-<ver>-cu<X>-x64.msi` | cu132, cu134 | `<ProgramFiles>\RiemannEngine\<ver>_cu<X>\` |
| `RiemannEngine-<ver>-cu<X>-win-x64.zip` | cu132, cu134 | `RiemannEngine\<ver>_cu<X>\{include,lib}` |
| `libriemann-cpp-dev_<ver>_cu<X>_ubuntu<U>_amd64.deb` | 22.04/cu124, 24.04/cu132, 24.04/cu134 | `/usr/local/RiemannEngine/<ver>_cu<X>/` |
| `RiemannEngine-<ver>-cu<X>-ubuntu<U>-x64.tar.gz` | same as .deb | `RiemannEngine/<ver>_cu<X>/{include,lib}` |
| Docker `acaicia/riemannengine:<ver>-cu<X>` and `:cu<X>` | cu124, cu132, cu134 | `:latest` also points at cu134 |
| `SHA256SUMS.txt` | | checksums of all files above |

The zips/tarballs contain only RiemannEngine (headers, static library, CMake
package files). Programs also need the matching LibTorch and CUDA toolkit.
Every version/CUDA combination installs side by side, so you choose one at
configure time:

```powershell
cmake -B build -DCMAKE_PREFIX_PATH="C:\Program Files\RiemannEngine\1.0.0_cu134;D:\libtorch_cu134"
```

| Parameter | Default | Description |
|---|---|---|
| `-Version` | from `CMakeLists.txt` | Release version |
| `-WindowsCuda` | `132,134` | Windows CUDA variants (LibTorch from `D:\libtorch_cu<X>`) |
| `-DebCuda` | `124,132,134` | Ubuntu .deb variants |
| `-DockerCuda` | `124,132,134` | Docker images |
| `-OutputDir` | `dist\<version>` | Artifact directory |
| `-Jobs` | `12` | Parallel compile jobs |
| `-CudaArchitectures` | `86` | GPU architectures for the Linux builds, e.g. `"86;89;90"` |
| `-DockerUser` / `-DockerRepo` | `$env:DOCKER_USER` or `acaicia` / `riemannengine` | Docker Hub image name |
| `-SkipWindows`, `-SkipDeb`, `-SkipDocker` | | Skip a whole stage |
| `-NoPush` | | Build Docker images without pushing |
| `-Clean` | | Wipe `build-release-cu<X>` directories first |
| `-StopOnError` | | Abort at the first failure (default: continue and report) |

Requirements: everything for `build-all-windows.ps1` (below), WiX v4+ with the
UI extension (see [msi/README.md](../msi/README.md)), Docker Desktop in Linux
containers mode, and `docker login` for pushing.

## Prerequisites

1. **Visual Studio 2022** with C++ development tools
2. **CUDA Toolkit 13.2 or 13.4**
3. **CMake 3.18+**
4. **Ninja build system**
5. **LibTorch** (PyTorch C++ distribution)

## Quick Start

### Option 1: All-in-One Build (every CUDA variant)

```powershell
.\scripts\build-all-windows.ps1 -Clean
```

With no arguments this builds every supported variant, each in its own process:

| Platform | CUDA | Output |
|---|---|---|
| Windows (VS 2022 + Ninja) | 13.2, 13.4 | `build-cu132`, `build-cu134` |
| Linux (Docker, `apt\build-apt.ps1`) | Ubuntu 22.04/12.4, 24.04/13.2, 24.04/13.4 | `.deb` + SDK tarball in `dist\<version>` |

A failed variant does not stop the others; a summary table is printed at the
end and the script exits non-zero if anything failed. The Linux variants need
Docker Desktop running (Linux containers).

### Option 2: Step-by-Step

```powershell
# 1. Configure
.\scripts\configure-windows.ps1 -Clean

# 2. Build
.\scripts\build-windows.ps1

# 3. Install (optional)
cmake --install build --config Release --prefix "C:\Program Files\RiemannEngine"
```

## Script Details

### `configure-windows.ps1`
Configures the CMake build system.

**Parameters:**
- `-LibTorchPath` - Path to LibTorch (default: `E:\Code\libtorch`)
- `-BuildDir` - Build directory (default: `build`)
- `-BuildType` - Build configuration (default: `Release`)
- `-CudaPath` - CUDA installation path
- `-NinjaPath` - Path to ninja.exe
- `-CpuOnly` - Build without CUDA support
- `-Clean` - Clean build directory before configuring

**Example:**
```powershell
.\scripts\configure-windows.ps1 `
  -LibTorchPath "D:\lib\libtorch" `
  -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" `
  -Clean
```

### `build-windows.ps1`
Builds the project (run after configure).

**Parameters:**
- `-BuildDir` - Build directory (default: `build`)
- `-BuildType` - Build configuration (default: `Release`)
- `-Jobs` - Number of parallel jobs (default: auto)
- `-Verbose` - Verbose build output
- `-Target` - Specific target to build

**Example:**
```powershell
.\scripts\build-windows.ps1 -Jobs 8 -Verbose
```

### `build-all-windows.ps1`
Complete build pipeline for every variant: configure + build (+ install) on
Windows, and `.deb`/tarball builds for Linux through Docker. It enters the
Visual Studio 2022 developer environment by itself.

**Parameters:**
- `-CudaVersion` - Optional filter, e.g. `134` or `124,132`. Each version is
  built on every platform that supports it (Windows: 132/134,
  Linux: 124/132/134). Default: all.
- `-SkipWindows` / `-SkipLinux` - Skip a platform
- `-BuildDir` - Base build directory; variants use `<BuildDir>-cu<X>` (default: `build`)
- `-LibTorchRoot` - Folder holding `libtorch_cu132` and `libtorch_cu134` (default: `D:\`)
- `-LibTorchPath`, `-CudaPath`, `-CudnnPath` - Explicit paths (only with a single Windows variant)
- `-BuildType`, `-NinjaPath`, `-Jobs`, `-Verbose`, `-Clean`, `-CpuOnly`
- `-Install` - Install each Windows variant to `<InstallPrefix>\RiemannEngine\<version>_cu<X>`
- `-InstallPrefix` - Default: `<repo>\install`
- `-LinuxOutputDir` - Where the Linux packages go (default: `dist\<version>`)
- `-CudaArchitectures` - GPU architectures for the Linux builds (default: `86`)

**Examples:**
```powershell
.\scripts\build-all-windows.ps1                            # everything
.\scripts\build-all-windows.ps1 -SkipLinux -Install -InstallPrefix "C:\sdk"
.\scripts\build-all-windows.ps1 -CudaVersion 134           # cu134 on Windows + Linux
.\scripts\build-all-windows.ps1 -CudaVersion 132 -SkipLinux -LibTorchPath "E:\libtorch"
```

## Environment Setup

**IMPORTANT:** All scripts must be run from a Visual Studio Developer environment.

### Method 1: Use Developer PowerShell
Open "x64 Native Tools Command Prompt for VS 2022" from Start Menu

### Method 2: Initialize in Existing PowerShell
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```

### Method 3: Add to PowerShell Profile
Add this to your PowerShell profile to create an alias:
```powershell
function Enter-VSDevShell {
    & "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
}
Set-Alias vsdev Enter-VSDevShell
```

Then just run `vsdev` in any PowerShell session.

## Troubleshooting

### "Not in Visual Studio Developer environment"
Run from VS Developer PowerShell or initialize the environment first.

### "CMake not found"
Ensure CMake is installed and in PATH. Visual Studio 2022 includes CMake.

### "Ninja not found"
Install Ninja and specify path with `-NinjaPath` parameter, or add to PATH.

### "LibTorch not found"
Download LibTorch from pytorch.org and specify correct path with `-LibTorchPath`.

### "CUDA not found"
Either install CUDA Toolkit, use `-CpuOnly` flag, or specify correct path with `-CudaPath`.

## CPU-Only Build

To build without CUDA (a single Windows build in `build-cpu`):

```powershell
.\scripts\build-all-windows.ps1 -CpuOnly -Clean
```

## See Also

- `build-notes.md` - Detailed technical notes about the build process
- `INSTALL.md` - General installation instructions
