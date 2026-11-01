# =============================================================================
# RiemannEngine C++ — Debian Package Builder (Windows host)
# =============================================================================
#
# Builds a .deb package for the selected CUDA/Ubuntu profile by spinning up a
# Docker build and exporting the finished package to the local machine.
#
# Prerequisites
# -------------
#   - Docker Desktop (Linux containers mode, BuildKit enabled)
#     BuildKit is on by default in Docker Desktop 23+.
#
# Usage
# -----
#   .\apt\build-apt.ps1
#   .\apt\build-apt.ps1 -Version 1.0.0 -Jobs 8 -OutputDir .\dist
#
# =============================================================================

[CmdletBinding()]
param(
    # Semver string embedded in the .deb filename and package metadata.
    # Defaults to project(VERSION) from CMakeLists.txt.
    [ValidatePattern('^(\d+\.\d+\.\d+)?$')]
    [string]$Version = "",

    # Parallel compile jobs passed to ninja inside the container.
    [ValidateRange(1, 1024)]
    [int]$Jobs = 4,

    [ValidateSet("124", "132", "134")]
    [string]$CudaVersion = "134",

    [string]$CudaArchitectures = "86",

    # Where to write the finished .deb.  Defaults to the apt\ directory.
    [string]$OutputDir = "",

    # Docker image tag used for the build cache; does not appear in the output.
    [string]$ImageTag = "",

    # Keep the intermediate Docker image after the build completes.
    # By default the image is removed to save disk space.
    [switch]$KeepImage
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$Dockerfile  = Join-Path $ScriptDir "Dockerfile"

if (-not $OutputDir) {
    $OutputDir = $ScriptDir
}
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path (Get-Location) $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $Version) {
    $cmakeLists = Get-Content (Join-Path $ProjectRoot "CMakeLists.txt") -Raw
    if ($cmakeLists -notmatch 'project\s*\([^)]*VERSION\s+(\d+\.\d+\.\d+)') {
        Write-Host "ERROR: Could not read project VERSION from CMakeLists.txt; pass -Version." -ForegroundColor Red
        exit 1
    }
    $Version = $Matches[1]
}

# Must match the profiles in docker/cuda-profile.sh.
$UbuntuVersion = @{ "124" = "22.04"; "132" = "24.04"; "134" = "24.04" }[$CudaVersion]

if (-not $ImageTag) { $ImageTag = "riemannengine-deb-builder:cu$CudaVersion" }
$ExpectedDeb = Join-Path $OutputDir "libriemann-cpp-dev_${Version}_cu${CudaVersion}_ubuntu${UbuntuVersion}_amd64.deb"
$ExpectedTarball = Join-Path $OutputDir "RiemannEngine-${Version}-cu${CudaVersion}-ubuntu${UbuntuVersion}-x64.tar.gz"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " RiemannEngine C++ - Debian Package Builder" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Version      : $Version"
Write-Host "  Jobs         : $Jobs"
Write-Host "  CUDA         : cu$CudaVersion (Ubuntu $UbuntuVersion)"
Write-Host "  Install dir  : /usr/local/RiemannEngine/${Version}_cu$CudaVersion"
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Dockerfile   : $Dockerfile"
Write-Host "  Output dir   : $OutputDir"
Write-Host "  Output file  : $(Split-Path -Leaf $ExpectedDeb)"
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

Write-Host "[1/3] Checking prerequisites..." -ForegroundColor Cyan

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'docker' not found on PATH." -ForegroundColor Red
    Write-Host "Install Docker Desktop (Linux containers mode) and try again." -ForegroundColor Yellow
    exit 1
}

# Verify Docker daemon is reachable
try {
    docker info --format "{{.ServerVersion}}" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "ERROR: Docker daemon is not running or not accessible." -ForegroundColor Red
    Write-Host "Start Docker Desktop and ensure Linux container mode is active." -ForegroundColor Yellow
    exit 1
}

Write-Host "  Docker OK" -ForegroundColor Green

if (-not (Test-Path $Dockerfile)) {
    Write-Host "ERROR: Dockerfile not found at: $Dockerfile" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Build the Docker image and export the .deb via BuildKit --output
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "[2/3] Building Docker image and packaging .deb..." -ForegroundColor Cyan
Write-Host "  This will download LibTorch (~2 GB) on the first run." -ForegroundColor DarkGray
Write-Host "  Subsequent runs reuse the Docker layer cache." -ForegroundColor DarkGray
Write-Host ""

# Force BuildKit (required for --output)
$env:DOCKER_BUILDKIT = "1"

# Build the 'artifact' target stage and export its filesystem to $OutputDir.
# The artifact stage is FROM scratch and contains only /libriemann-cpp-dev_*.deb.
$buildArgs = @(
    "build",
    "--file", $Dockerfile,
    "--platform", "linux/amd64",
    "--build-arg", "VERSION=$Version",
    "--build-arg", "JOBS=$Jobs",
    "--build-arg", "CUDA_VERSION=$CudaVersion",
    "--build-arg", "CUDA_ARCHITECTURES=$CudaArchitectures",
    "--target", "artifact",
    "--output", "type=local,dest=$OutputDir",
    "--tag", $ImageTag,
    $ProjectRoot
)

Write-Host "  Running: docker $($buildArgs -join ' ')`n" -ForegroundColor DarkGray
& docker @buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Docker build failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Verify output and report
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "[3/3] Verifying output..." -ForegroundColor Cyan

if (-not (Test-Path $ExpectedDeb)) {
    Write-Host "ERROR: Expected .deb not found: $ExpectedDeb" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $ExpectedTarball)) {
    Write-Host "ERROR: Expected SDK tarball not found: $ExpectedTarball" -ForegroundColor Red
    exit 1
}

$sizeMb = [math]::Round((Get-Item $ExpectedDeb).Length / 1MB, 2)
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host " Build complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Package : $(Split-Path -Leaf $ExpectedDeb)"
Write-Host "  SDK     : $(Split-Path -Leaf $ExpectedTarball)"
Write-Host "  Size    : $sizeMb MB"
Write-Host "  Path    : $ExpectedDeb"
Write-Host ""
Write-Host "  To install on a target machine with matching Ubuntu, CUDA and LibTorch:" -ForegroundColor DarkGray
Write-Host "    sudo dpkg -i $(Split-Path -Leaf $ExpectedDeb)" -ForegroundColor DarkGray
Write-Host "    cmake -DCMAKE_PREFIX_PATH=`"/usr/local/RiemannEngine/${Version}_cu$CudaVersion;/path/to/libtorch`" .." -ForegroundColor DarkGray
Write-Host "================================================================" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Clean up intermediate image (opt-out with -KeepImage)
# ---------------------------------------------------------------------------

if (-not $KeepImage) {
    # buildx --output to a local dir usually doesn't load the image, so absence is fine.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    docker image inspect $ImageTag *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "  Removing intermediate build image '$ImageTag'..." -ForegroundColor DarkGray
        docker rmi $ImageTag *> $null
    }
    $ErrorActionPreference = $prevEap
}
exit 0
