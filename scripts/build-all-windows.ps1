
# RiemannEngine C++ - Complete Build (every CUDA variant, Windows + Linux)
#
# By default this builds every supported variant:
#   Windows (native, VS 2022 + Ninja): CUDA 13.2, 13.4
#       -> <BuildDir>-cu132, <BuildDir>-cu134
#   Linux (Docker, via apt\build-apt.ps1): Ubuntu 22.04/CUDA 12.4,
#       Ubuntu 24.04/CUDA 13.2, Ubuntu 24.04/CUDA 13.4
#       -> .deb + SDK tarball in -LinuxOutputDir (default dist\<version>)
# Each variant runs in its own PowerShell process, so CUDA/VS environment
# changes never leak between variants. A failed variant does not stop the
# others; the script exits non-zero if any variant failed.
#
# Examples:
#   .\scripts\build-all-windows.ps1                        # everything
#   .\scripts\build-all-windows.ps1 -Clean
#   .\scripts\build-all-windows.ps1 -SkipLinux             # Windows variants only
#   .\scripts\build-all-windows.ps1 -SkipWindows           # Linux variants only
#   .\scripts\build-all-windows.ps1 -CudaVersion 134       # cu134 on Windows and Linux
#   .\scripts\build-all-windows.ps1 -CudaVersion 132,134 -SkipLinux
#   .\scripts\build-all-windows.ps1 -Install -InstallPrefix "D:\sdk"
#       -> D:\sdk\RiemannEngine\<version>_cu<X> per Windows variant
#   .\scripts\build-all-windows.ps1 -CpuOnly               # one CPU-only Windows build
#
# Windows variant N uses the CUDA toolkit in
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x and LibTorch in
# D:\libtorch_cu13x. Override them with -CudaPath / -LibTorchPath (only when a
# single Windows variant is selected) or -LibTorchRoot (D:\ by default).
# Linux variants need Docker Desktop running in Linux-containers mode.

param(
    # Filter: build only these CUDA versions (e.g. 134 or 124,132). Windows
    # supports 132/134, Linux 124/132/134; each version is built wherever
    # it is supported.
    [string[]]$CudaVersion = @(),
    [string]$LibTorchPath = "",
    [string]$LibTorchRoot = "D:\",
    # Base build directory; each Windows variant uses <BuildDir>-cu<X>.
    [string]$BuildDir = "build",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Release",
    # Each Windows variant installs to <InstallPrefix>\RiemannEngine\<version>_cu<X>
    # (default prefix: <repo>\install).
    [string]$InstallPrefix = "",
    [string]$CudaPath = "",
    [string]$CudnnPath = "",
    [string]$NinjaPath = "C:\Program Files\Ninja\ninja.exe",
    [string]$LinuxOutputDir = "",
    [string]$CudaArchitectures = "86",
    [switch]$SkipWindows,
    [switch]$SkipLinux,
    [switch]$CpuOnly,
    [switch]$Clean,
    [switch]$Install,
    [switch]$Verbose,
    [int]$Jobs = 12,
    # Internal: build exactly one Windows variant in this process.
    [ValidateSet("", "132", "134")]
    [string]$Variant = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot

$WindowsCudaAll = @("132", "134")
$LinuxCudaAll = @("124", "132", "134")

# ------------------------------------------------------------
# Orchestrator: fan out to one process per variant
# ------------------------------------------------------------

if (-not $Variant) {
    $cmakeLists = Get-Content (Join-Path $ProjectRoot "CMakeLists.txt") -Raw
    if ($cmakeLists -notmatch 'project\s*\([^)]*VERSION\s+(\d+\.\d+\.\d+)') {
        Write-Host "[ERROR] Could not read project VERSION from CMakeLists.txt." -ForegroundColor Red
        exit 1
    }
    $projectVersion = $Matches[1]

    $requested = @($CudaVersion | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ } |
        ForEach-Object { $_.Replace('.', '') } | Select-Object -Unique)
    $unknown = @($requested | Where-Object { ($WindowsCudaAll + $LinuxCudaAll) -notcontains $_ })
    if ($unknown.Count) {
        Write-Host "[ERROR] Unsupported -CudaVersion: $($unknown -join ', '). Use 124, 132 or 134." -ForegroundColor Red
        exit 1
    }
    $windowsVariants = @(if ($SkipWindows) { } elseif ($requested.Count) { $WindowsCudaAll | Where-Object { $requested -contains $_ } } else { $WindowsCudaAll })
    $linuxVariants = @(if ($SkipLinux -or $CpuOnly) { } elseif ($requested.Count) { $LinuxCudaAll | Where-Object { $requested -contains $_ } } else { $LinuxCudaAll })

    if ($CpuOnly -and $windowsVariants.Count -gt 1) {
        # A CPU-only build is the same for every CUDA version; build it once.
        $windowsVariants = @($windowsVariants[-1])
    }

    if ($windowsVariants.Count -eq 0 -and $linuxVariants.Count -eq 0) {
        Write-Host "[ERROR] Nothing to build for -CudaVersion $($CudaVersion -join ',')." -ForegroundColor Red
        exit 1
    }

    if ($windowsVariants.Count -gt 1 -and ($CudaPath -or $LibTorchPath)) {
        Write-Host "[ERROR] -CudaPath/-LibTorchPath apply to one variant; add -CudaVersion <X> or use -LibTorchRoot." -ForegroundColor Red
        exit 1
    }

    if (-not $LinuxOutputDir) { $LinuxOutputDir = Join-Path $ProjectRoot "dist\$projectVersion" }
    if ($Install -and -not $InstallPrefix) { $InstallPrefix = Join-Path $ProjectRoot "install" }

    $hostExe = (Get-Process -Id $PID).Path
    $results = [System.Collections.Generic.List[object]]::new()

    function Invoke-Variant {
        param([string]$Name, [string]$Script, [string[]]$Arguments)

        Write-Host ""
        Write-Host ("#" * 100) -ForegroundColor Magenta
        Write-Host "# $Name" -ForegroundColor Magenta
        Write-Host ("#" * 100) -ForegroundColor Magenta

        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        & $hostExe -NoProfile -ExecutionPolicy Bypass -File $Script @Arguments
        $code = $LASTEXITCODE
        $timer.Stop()

        $results.Add([pscustomobject]@{
            Variant  = $Name
            Status   = $(if ($code -eq 0) { "OK" } else { "FAILED (exit $code)" })
            Duration = $timer.Elapsed.ToString("hh\:mm\:ss")
        })
    }

    Write-Host ""
    Write-Host "RiemannEngine $projectVersion - building all variants" -ForegroundColor Cyan
    Write-Host "  Windows : $(if ($windowsVariants.Count) { ($windowsVariants | ForEach-Object { if ($CpuOnly) { 'cpu' } else { "cu$_" } }) -join ', ' } else { 'skipped' })"
    Write-Host "  Linux   : $(if ($linuxVariants.Count) { ($linuxVariants | ForEach-Object { "cu$_" }) -join ', ' } else { 'skipped' })"

    foreach ($cuda in $windowsVariants) {
        $suffix = if ($CpuOnly) { "cpu" } else { "cu$cuda" }
        $childArgs = @(
            "-Variant", $cuda,
            "-BuildDir", "$BuildDir-$suffix",
            "-BuildType", $BuildType,
            "-LibTorchRoot", $LibTorchRoot,
            "-NinjaPath", $NinjaPath,
            "-Jobs", "$Jobs"
        )
        if ($LibTorchPath) { $childArgs += @("-LibTorchPath", $LibTorchPath) }
        if ($CudaPath) { $childArgs += @("-CudaPath", $CudaPath) }
        if ($CudnnPath) { $childArgs += @("-CudnnPath", $CudnnPath) }
        if ($Install) {
            $childArgs += @("-Install", "-InstallPrefix", (Join-Path $InstallPrefix "RiemannEngine\${projectVersion}_$suffix"))
        }
        foreach ($flag in "CpuOnly", "Clean", "Verbose") {
            if ((Get-Variable $flag -ValueOnly).IsPresent) { $childArgs += "-$flag" }
        }
        Invoke-Variant "Windows $suffix" $PSCommandPath $childArgs
    }

    if ($linuxVariants.Count) {
        $dockerReady = $false
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            & docker info --format "{{.ServerVersion}}" *> $null
            $dockerReady = ($LASTEXITCODE -eq 0)
        }

        foreach ($cuda in $linuxVariants) {
            if (-not $dockerReady) {
                $results.Add([pscustomobject]@{
                    Variant  = "Linux cu$cuda"
                    Status   = "FAILED (Docker daemon not reachable; start Docker Desktop)"
                    Duration = "00:00:00"
                })
                continue
            }
            Invoke-Variant "Linux cu$cuda" (Join-Path $ProjectRoot "apt\build-apt.ps1") @(
                "-Version", $projectVersion,
                "-CudaVersion", $cuda,
                "-CudaArchitectures", $CudaArchitectures,
                "-Jobs", "$Jobs",
                "-OutputDir", $LinuxOutputDir
            )
        }
    }

    Write-Host ""
    Write-Host ("=" * 100) -ForegroundColor Cyan
    Write-Host "Build summary" -ForegroundColor Cyan
    Write-Host ("=" * 100) -ForegroundColor Cyan
    $results | Format-Table Variant, Status, Duration -AutoSize | Out-String -Width 200 | Write-Host
    if ($linuxVariants.Count) { Write-Host "Linux packages: $LinuxOutputDir" }
    if ($Install) { Write-Host "Windows SDKs:   $(Join-Path $InstallPrefix 'RiemannEngine')" }

    $failed = @($results | Where-Object { $_.Status -ne "OK" })
    if ($failed.Count) {
        Write-Host "$($failed.Count) variant(s) failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "All variants built successfully." -ForegroundColor Green
    exit 0
}

# ------------------------------------------------------------
# Single Windows variant (-Variant <X>)
# ------------------------------------------------------------

$cudaTag = $Variant
$cudaDotted = "$($cudaTag.Substring(0, 2)).$($cudaTag.Substring(2))"

if (-not $CudaPath) {
    $CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaDotted"
}

if (-not $LibTorchPath) {
    $LibTorchPath = Join-Path $LibTorchRoot "libtorch_cu$cudaTag"
}

function Write-Section {
    param([string]$Title)

    Write-Host ""
    Write-Host ("=" * 100) -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * 100) -ForegroundColor Cyan
}

function Fail {
    param([string]$Message)

    Write-Host ""
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Resolve-ProjectPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot $Path)
    )
}

function Find-FirstFile {
    param(
        [string[]]$Roots,
        [string[]]$Names,
        [string[]]$PathPreference = @()
    )

    foreach ($root in $Roots) {
        if (-not $root -or -not (Test-Path -LiteralPath $root -PathType Container)) {
            continue
        }

        foreach ($name in $Names) {
            $direct = Join-Path $root $name

            if (Test-Path -LiteralPath $direct -PathType Leaf) {
                return (Get-Item -LiteralPath $direct).FullName
            }
        }

        $matches = @()
        foreach ($name in $Names) {
            $matches += Get-ChildItem -LiteralPath $root `
                -Filter $name `
                -File `
                -Recurse `
                -ErrorAction SilentlyContinue
        }

        # Earlier patterns win (exact CUDA version, then same major version).
        foreach ($pattern in $PathPreference) {
            $preferred = $matches | Where-Object { $_.FullName -match $pattern } | Select-Object -First 1
            if ($preferred) {
                return $preferred.FullName
            }
        }

        if ($PathPreference.Count -eq 0 -and $matches) {
            return $matches[0].FullName
        }
    }

    return $null
}

function Get-CudaVersion {
    param([string]$Nvcc)

    $output = (& $Nvcc --version 2>&1) -join "`n"

    if ($LASTEXITCODE -ne 0) {
        Fail "Unable to execute nvcc: $Nvcc"
    }

    if ($output -match 'release\s+(\d+\.\d+)') {
        return $Matches[1]
    }

    Fail "Unable to determine CUDA version from nvcc output."
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$Description
    )

    Write-Host ""
    Write-Host $Description -ForegroundColor Cyan
    Write-Host "$Executable $($Arguments -join ' ')" -ForegroundColor DarkGray
    Write-Host ""

    & $Executable @Arguments

    $code = $LASTEXITCODE

    if ($code -ne 0) {
        Write-Host ""
        Write-Host "[FAILED] $Description (exit code $code)" -ForegroundColor Red
        exit $code
    }
}

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

Write-Section "RiemannEngine C++ - Complete Windows Build"

if ($true) {
    Write-Host "Entering Visual Studio 2022 Developer environment..." -ForegroundColor Yellow

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

    if (-not (Test-Path -LiteralPath $vswhere)) {
        Fail "vswhere.exe not found. Install Visual Studio 2022 with the C++ workload."
    }

    # CUDA 13.x supports VS 2022 (17.x); newer VS releases are rejected by nvcc.
    $vsRoot = (& $vswhere -version "[17.0,18.0)" -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1)

    if (-not $vsRoot) {
        Fail "Visual Studio 2022 with the C++ tools was not found (required by CUDA 13.x)."
    }

    $launch = if ($vsRoot) { Join-Path $vsRoot "Common7\Tools\Launch-VsDevShell.ps1" } else { $null }

    if (-not $launch -or -not (Test-Path -LiteralPath $launch)) {
        Fail "Visual Studio C++ tools not found. Install the 'Desktop development with C++' workload."
    }

    # Drop any inherited (e.g. VS 18) toolchain environment so INCLUDE/LIB/PATH
    # come only from VS 2022; otherwise the newer STL headers are still used.
    Get-ChildItem Env: |
        Where-Object { $_.Name -match '^(VSCMD_.*|VSINSTALLDIR|VCINSTALLDIR|VCToolsInstallDir|VCToolsVersion|VCIDEInstallDir|VisualStudioVersion|INCLUDE|LIB|LIBPATH|EXTERNAL_INCLUDE|UCRTVersion|UniversalCRTSdkDir|WindowsSDK.*|DevEnvDir|Platform|__VSCMD.*)$' } |
        ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" -ErrorAction SilentlyContinue }
    $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -and $_ -notmatch 'Microsoft Visual Studio\\(18|2026)' } | Select-Object -Unique) -join ';'

    # Launch-VsDevShell changes the working directory.
    Push-Location
    try {
        & $launch -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
    } finally {
        Pop-Location
    }

    # An existing VS 18 shell can leave its cl.exe first on PATH; make sure 2022 wins.
    $vs2022Cl = Get-ChildItem -Path (Join-Path $vsRoot "VC\Tools\MSVC") -Directory |
        Sort-Object Name -Descending | Select-Object -First 1
    $clDir = Join-Path $vs2022Cl.FullName "bin\Hostx64\x64"
    $env:PATH = "$clDir;$env:PATH"

    $activeCl = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source

    if (-not $activeCl -or -not $activeCl.StartsWith($vsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Fail "Failed to select the Visual Studio 2022 compiler (active cl.exe: $activeCl)."
    }

    Write-Host "[OK] Host compiler: $activeCl" -ForegroundColor Green

    if ($env:INCLUDE -match 'Microsoft Visual Studio\\(18|2026)') {
        Fail "INCLUDE still references a newer Visual Studio (not supported by CUDA 13.x)."
    }
}

$hostCl = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source

if (-not $hostCl) {
    Fail "cl.exe not found on PATH."
}

$hostCl = $hostCl.Replace('\', '/')

if ($Jobs -lt 1) {
    Fail "Jobs must be at least 1."
}

$BuildDir = Resolve-ProjectPath $BuildDir
$LibTorchPath = Resolve-ProjectPath $LibTorchPath

if (-not (Test-Path -LiteralPath $LibTorchPath -PathType Container)) {
    Fail "LibTorch directory does not exist: $LibTorchPath"
}

$torchConfig = Join-Path $LibTorchPath "share\cmake\Torch\TorchConfig.cmake"

if (-not (Test-Path -LiteralPath $torchConfig -PathType Leaf)) {
    Fail "TorchConfig.cmake not found: $torchConfig"
}

$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue

if (-not $cmakeCommand) {
    Fail "CMake is not installed or not available in PATH."
}

$CmakeExe = $cmakeCommand.Source

if (-not (Test-Path -LiteralPath $NinjaPath -PathType Leaf)) {
    $ninjaCommand = Get-Command ninja -ErrorAction SilentlyContinue

    if (-not $ninjaCommand) {
        Fail "Ninja not found: $NinjaPath"
    }

    $NinjaPath = $ninjaCommand.Source
}

$NinjaPath = (Get-Item -LiteralPath $NinjaPath).FullName

# ------------------------------------------------------------
# CUDA configuration
# ------------------------------------------------------------

$cudaArgs = @()
$cudnnArgs = @()

if (-not $CpuOnly) {

    Write-Section "CUDA Detection"

    $CudaPath = Resolve-ProjectPath $CudaPath
    $Nvcc = Join-Path $CudaPath "bin\nvcc.exe"

    if (-not (Test-Path -LiteralPath $Nvcc -PathType Leaf)) {
        Fail "CUDA compiler not found: $Nvcc"
    }

    # Forward slashes avoid "Invalid character escape" errors in Torch's FindCUDA.
    $CudaPath = $CudaPath.Replace('\', '/')
    $Nvcc = $Nvcc.Replace('\', '/')

    $cudaVersion = Get-CudaVersion $Nvcc

    Write-Host "[OK] CUDA version: $cudaVersion" -ForegroundColor Green
    Write-Host "     CUDA root: $CudaPath"

    # Ensure CUDA detection is consistent across CMake and Caffe2.
    $env:CUDA_PATH = $CudaPath
    $env:CUDA_HOME = $CudaPath
    $env:CUDAToolkit_ROOT = $CudaPath
    $env:CUDACXX = $Nvcc

    # Put the selected toolkit first and drop other CUDA versions from PATH.
    $otherCuda = '\\NVIDIA GPU Computing Toolkit\\CUDA\\v\d+\.\d+\\'
    $cleanPath = ($env:PATH -split ';' | Where-Object {
        $_ -and (($_ -notmatch $otherCuda) -or ($_.Replace('\', '/') -like "$CudaPath*"))
    }) -join ';'
    $env:PATH = "$CudaPath/bin;$CudaPath/bin/x64;$cleanPath"

    $cudaArgs = @(
        "-DCUDAToolkit_ROOT=$CudaPath"
        "-DCUDA_TOOLKIT_ROOT_DIR=$CudaPath"
        "-DCMAKE_CUDA_COMPILER=$Nvcc"
        "-DCUDA_NVCC_EXECUTABLE=$Nvcc"
    )

    # --------------------------------------------------------
    # cuDNN detection
    # --------------------------------------------------------

    Write-Section "cuDNN Detection"

    $cudnnRoots = @()

    if ($CudnnPath) {
        $cudnnRoots += Resolve-ProjectPath $CudnnPath
    }

    if ($env:CUDNN_ROOT) {
        $cudnnRoots += $env:CUDNN_ROOT
    }

    if ($env:CUDNN_PATH) {
        $cudnnRoots += $env:CUDNN_PATH
    }

    $cudnnRoots += @(
        $CudaPath
        "C:\Program Files\NVIDIA\CUDNN"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDNN"
        "C:\tools\cuda"
    )

    $cudnnRoots = @(
        $cudnnRoots |
        Where-Object { $_ } |
        Select-Object -Unique
    )

    $cudaMajor = $cudaDotted.Split('.')[0]
    $cudnnPref = @(
        "[\\/]$([regex]::Escape($cudaDotted))[\\/]",
        "[\\/]$cudaMajor\.\d+[\\/]"
    )

    $cudnnHeader = Find-FirstFile `
        -Roots $cudnnRoots `
        -Names @("cudnn.h") `
        -PathPreference $cudnnPref

    $cudnnLibrary = Find-FirstFile `
        -Roots $cudnnRoots `
        -Names @("cudnn.lib", "cudnn64_*.lib") `
        -PathPreference $cudnnPref

    $cudnnRuntime = Find-FirstFile `
        -Roots $cudnnRoots `
        -Names @("cudnn64_*.dll") `
        -PathPreference $cudnnPref

    if (-not $cudnnHeader -or -not $cudnnLibrary) {

        Write-Host "cuDNN development files were not found." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Checked locations:"
        $cudnnRoots | ForEach-Object { Write-Host "  $_" }

        Fail "Install compatible cuDNN or specify -CudnnPath."
    }

    $cudnnInclude = Split-Path -Parent $cudnnHeader
    $cudnnBin = if ($cudnnRuntime) {
        Split-Path -Parent $cudnnRuntime
    } else {
        $null
    }

    Write-Host "[OK] cuDNN header: $cudnnHeader" -ForegroundColor Green
    Write-Host "[OK] cuDNN library: $cudnnLibrary" -ForegroundColor Green

    if ($cudnnRuntime) {
        Write-Host "[OK] cuDNN runtime: $cudnnRuntime" -ForegroundColor Green
        $env:PATH = "$cudnnBin;$env:PATH"
    } else {
        Write-Host "[WARNING] cuDNN runtime DLL not found." -ForegroundColor Yellow
    }

    # Torch's cuDNN check reads CUDNN_MAJOR from cudnn.h, but cuDNN 9 keeps the
    # version macros in cudnn_version.h. Generate a shim cudnn.h that exposes
    # them and includes the real header.
    $cudnnHasVersion = Select-String -LiteralPath $cudnnHeader -Pattern '#\s*define\s+CUDNN_MAJOR' -Quiet

    if (-not $cudnnHasVersion) {
        $versionHeader = Join-Path $cudnnInclude "cudnn_version.h"

        if (-not (Test-Path -LiteralPath $versionHeader)) {
            Fail "cuDNN version header not found: $versionHeader"
        }

        $versionText = Get-Content -LiteralPath $versionHeader -Raw
        $parts = foreach ($macro in "CUDNN_MAJOR", "CUDNN_MINOR", "CUDNN_PATCHLEVEL") {
            if ($versionText -notmatch "#\s*define\s+$macro\s+(\d+)") {
                Fail "$macro not found in $versionHeader"
            }
            "#define $macro $($Matches[1])"
        }

        $shimDir = Join-Path $env:TEMP "riemann_cudnn_shim_$cudaTag"
        New-Item -ItemType Directory -Path $shimDir -Force | Out-Null

        $realHeader = $cudnnHeader.Replace('\', '/')
        $shim = @("#pragma once", "#include `"$realHeader`"") + ($parts | ForEach-Object { "#ifndef $($_.Split(' ')[1])"; $_; "#endif" })
        Set-Content -LiteralPath (Join-Path $shimDir "cudnn.h") -Value $shim -Encoding ASCII

        Write-Host "[INFO] Using cuDNN version shim: $shimDir" -ForegroundColor Yellow
        $cudnnInclude = $shimDir
    }

    $env:CUDNN_INCLUDE_DIR = $cudnnInclude
    $env:CUDNN_LIBRARY = $cudnnLibrary

    $cudnnArgs = @(
        "-DCUDNN_INCLUDE_DIR=$cudnnInclude"
        "-DCUDNN_LIBRARY=$cudnnLibrary"
        "-DCUDNN_INCLUDE_PATH=$cudnnInclude"
        "-DCUDNN_LIBRARY_PATH=$cudnnLibrary"
    )
}

# ------------------------------------------------------------
# Clean / cache handling
# ------------------------------------------------------------

Write-Section "Build Directory"

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
    Write-Host "Removing: $BuildDir" -ForegroundColor Yellow
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

$cacheFile = Join-Path $BuildDir "CMakeCache.txt"

if (Test-Path -LiteralPath $cacheFile) {
    $cacheText = Get-Content -LiteralPath $cacheFile -Raw

    $cacheCuda = [regex]::Match(
        $cacheText,
        '(?m)^CMAKE_CUDA_COMPILER:FILEPATH=(.+)$'
    )

    if ($cacheCuda.Success -and -not $CpuOnly) {
        $cachedCompiler = $cacheCuda.Groups[1].Value.Trim()

        if ($cachedCompiler -ne $Nvcc) {
            Fail "CMake cache uses a different CUDA compiler ($cachedCompiler). Rerun with -Clean."
        }
    }
}

# ------------------------------------------------------------
# Configure
# ------------------------------------------------------------

Write-Section "STEP 1: Configuration"

Write-Host "Project:     $ProjectRoot"
Write-Host "Build:       $BuildDir"
Write-Host "Build type:  $BuildType"
Write-Host "LibTorch:    $LibTorchPath"
Write-Host "CPU only:    $CpuOnly"
Write-Host "Jobs:        $Jobs"

$configureArgs = @(
    "-S", $ProjectRoot,
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=$hostCl",
    "-DCMAKE_CXX_COMPILER=$hostCl",
    "-DCMAKE_PREFIX_PATH=$LibTorchPath",
    "-DCMAKE_MAKE_PROGRAM=$NinjaPath",
    "-Wno-dev",
    "-Wno-unused-cli"
)

if ($CpuOnly) {
    $configureArgs += "-DCPU_ONLY=ON"
} else {
    $configureArgs += "-DCPU_ONLY=OFF"
    $configureArgs += $cudaArgs
    $configureArgs += $cudnnArgs
}

if ($InstallPrefix) {
    $configureArgs += "-DCMAKE_INSTALL_PREFIX=$(Resolve-ProjectPath $InstallPrefix)"
}

Invoke-Checked `
    -Executable $CmakeExe `
    -Arguments $configureArgs `
    -Description "Configuring project"

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

Write-Section "STEP 2: Build"

$buildArgs = @(
    "--build", $BuildDir,
    "--config", $BuildType,
    "--parallel", "$Jobs"
)

if ($Verbose) {
    $buildArgs += "--verbose"
}

Invoke-Checked `
    -Executable $CmakeExe `
    -Arguments $buildArgs `
    -Description "Building RiemannEngine C++"

# ------------------------------------------------------------
# Install
# ------------------------------------------------------------

if ($Install) {

    Write-Section "STEP 3: Install"

    $installArgs = @(
        "--install", $BuildDir,
        "--config", $BuildType
    )

    if ($InstallPrefix) {
        $installArgs += @(
            "--prefix",
            (Resolve-ProjectPath $InstallPrefix)
        )
    }

    Invoke-Checked `
        -Executable $CmakeExe `
        -Arguments $installArgs `
        -Description "Installing RiemannEngine C++"
}

# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------

Write-Host ""
Write-Host ("=" * 100) -ForegroundColor Green
Write-Host "BUILD COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 100) -ForegroundColor Green
Write-Host ""

Write-Host "Build directory: $BuildDir" -ForegroundColor Green

if (-not $Install) {
    Write-Host ""
    Write-Host "To install:"
    Write-Host "  cmake --install `"$BuildDir`" --config $BuildType"
}

exit 0
