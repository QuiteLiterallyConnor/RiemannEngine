<#
.SYNOPSIS
    Builds every RiemannEngine release artifact and pushes the Docker images.

.DESCRIPTION
    For the release version (default: project(VERSION) in CMakeLists.txt):

      Windows (per CUDA 132/134)
        - builds with scripts\build-all-windows.ps1
        - RiemannEngine-<ver>-cu<X>-win-x64.zip   SDK: RiemannEngine\<ver>_cu<X>\{include,lib}
        - RiemannEngine-<ver>-cu<X>-x64.msi       installs to <ProgramFiles>\RiemannEngine\<ver>_cu<X>

      Ubuntu (22.04/cu124, 24.04/cu132, 24.04/cu134; built in Docker)
        - libriemann-cpp-dev_<ver>_cu<X>_ubuntu<U>_amd64.deb   installs to /usr/local/RiemannEngine/<ver>_cu<X>
        - RiemannEngine-<ver>-cu<X>-ubuntu<U>-x64.tar.gz        SDK with the same layout

      Docker (per CUDA 124/132/134)
        - <user>/<repo>:<ver>-cu<X> and <user>/<repo>:cu<X>, plus :latest for cu134

    Everything is written to dist\<ver>\ together with SHA256SUMS.txt.
    Failed steps are reported at the end and make the script exit non-zero;
    use -StopOnError to abort at the first failure instead.

.EXAMPLE
    .\scripts\build-release.ps1

.EXAMPLE
    .\scripts\build-release.ps1 -SkipDocker -WindowsCuda 134 -DebCuda 134

.EXAMPLE
    .\scripts\build-release.ps1 -NoPush -Clean -Jobs 16
#>

[CmdletBinding()]
param(
    [ValidatePattern('^(\d+\.\d+\.\d+)?$')]
    [string]$Version = "",

    [ValidateSet("132", "134")]
    [string[]]$WindowsCuda = @("132", "134"),

    [ValidateSet("124", "132", "134")]
    [string[]]$DebCuda = @("124", "132", "134"),

    [ValidateSet("124", "132", "134")]
    [string[]]$DockerCuda = @("124", "132", "134"),

    # Defaults to dist\<version> in the repository root.
    [string]$OutputDir = "",

    [ValidateRange(1, 1024)]
    [int]$Jobs = 12,

    # GPU architectures for the Linux (.deb and Docker) builds.
    [string]$CudaArchitectures = "86",

    [string]$DockerUser = $(if ($env:DOCKER_USER) { $env:DOCKER_USER } else { "acaicia" }),
    [string]$DockerRepo = $(if ($env:DOCKER_REPO) { $env:DOCKER_REPO } else { "riemannengine" }),

    [switch]$SkipWindows,
    [switch]$SkipDeb,
    [switch]$SkipDocker,

    # Build the Docker images but do not push them.
    [switch]$NoPush,

    # Wipe the Windows build directories before configuring.
    [switch]$Clean,

    [switch]$StopOnError
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$HostExe = (Get-Process -Id $PID).Path

if (-not $Version) {
    $cmakeLists = Get-Content (Join-Path $ProjectRoot "CMakeLists.txt") -Raw
    if ($cmakeLists -notmatch 'project\s*\([^)]*VERSION\s+(\d+\.\d+\.\d+)') {
        throw "Could not read project VERSION from CMakeLists.txt; pass -Version."
    }
    $Version = $Matches[1]
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "dist\$Version"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path (Get-Location) $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path
$StageRoot = Join-Path $OutputDir "_stage"

$Results = [System.Collections.Generic.List[object]]::new()

function Write-Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

# Each step runs in its own PowerShell process so environment changes made by
# the per-variant scripts (VS dev shell, CUDA paths) never leak between steps.
function Invoke-ChildScript {
    param([string]$Script, [string[]]$Arguments)
    & $HostExe -NoProfile -ExecutionPolicy Bypass -File $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$(Split-Path -Leaf $Script) failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Step {
    param([string]$Name, [scriptblock]$Action)
    Write-Banner $Name
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $status = "OK"
    $detail = ""
    try {
        & $Action | Out-Host
    } catch {
        $status = "FAILED"
        $detail = $_.Exception.Message
        Write-Host "[FAILED] ${Name}: $detail" -ForegroundColor Red
    }
    $timer.Stop()
    $Results.Add([pscustomobject]@{
        Step     = $Name
        Status   = $status
        Duration = $timer.Elapsed.ToString("hh\:mm\:ss")
        Detail   = $detail
    })
    if ($status -ne "OK" -and $StopOnError) {
        throw "Stopping after failed step: $Name"
    }
    return ($status -eq "OK")
}

function Assert-Command {
    param([string]$Name, [string]$Hint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "'$Name' not found on PATH. $Hint"
    }
}

# -- Preflight ----------------------------------------------------------------

Write-Banner "RiemannEngine release $Version"
Write-Host "  Output      : $OutputDir"
Write-Host "  Windows     : $(if ($SkipWindows) { 'skipped' } else { ($WindowsCuda | ForEach-Object { "cu$_" }) -join ', ' })"
Write-Host "  Debian      : $(if ($SkipDeb) { 'skipped' } else { ($DebCuda | ForEach-Object { "cu$_" }) -join ', ' })"
Write-Host "  Docker      : $(if ($SkipDocker) { 'skipped' } else { "$DockerUser/${DockerRepo}: " + (($DockerCuda | ForEach-Object { "cu$_" }) -join ', ') + $(if ($NoPush) { ' (no push)' } else { '' }) })"

if (-not $SkipWindows) {
    Assert-Command "cmake" "Install CMake and add it to PATH."
    Assert-Command "wix" "Install WiX v4: dotnet tool install --global wix"
}
if (-not ($SkipDeb -and $SkipDocker)) {
    Assert-Command "docker" "Install Docker Desktop (Linux containers)."
    & docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) { throw "Docker daemon is not reachable. Start Docker Desktop." }
}
if (-not $SkipDocker -and -not $NoPush -and -not $DockerUser) {
    throw "Pushing requires a Docker Hub namespace: pass -DockerUser or set DOCKER_USER."
}

# -- Windows: build, SDK zip, MSI ---------------------------------------------

if (-not $SkipWindows) {
    foreach ($cuda in $WindowsCuda) {
        $buildDir = Join-Path $ProjectRoot "build-release-cu$cuda"
        $variantStage = Join-Path $StageRoot "win-cu$cuda"
        $sdkDir = Join-Path $variantStage "RiemannEngine\${Version}_cu$cuda"

        $built = Invoke-Step "Windows cu$cuda - build" {
            $buildArgs = @("-CudaVersion", $cuda, "-SkipLinux", "-BuildDir", (Join-Path $ProjectRoot "build-release"), "-Jobs", "$Jobs")
            if ($Clean) { $buildArgs += "-Clean" }
            Invoke-ChildScript (Join-Path $PSScriptRoot "build-all-windows.ps1") $buildArgs
        }
        if (-not $built) { continue }

        $zipped = Invoke-Step "Windows cu$cuda - SDK zip" {
            if (Test-Path $variantStage) { Remove-Item -Recurse -Force $variantStage }
            Invoke-Native "cmake" @("--install", $buildDir, "--prefix", $sdkDir, "--config", "Release")
            $zip = Join-Path $OutputDir "RiemannEngine-$Version-cu$cuda-win-x64.zip"
            if (Test-Path $zip) { Remove-Item -Force $zip }
            # bsdtar writes portable forward-slash entry names (Compress-Archive on PS 5.1 does not).
            Invoke-Native "tar.exe" @("-a", "-c", "-f", $zip, "-C", $variantStage, "RiemannEngine")
            Write-Host "  Created $zip" -ForegroundColor Green
        }
        if (-not $zipped) { continue }

        [void](Invoke-Step "Windows cu$cuda - MSI" {
            Invoke-ChildScript (Join-Path $ProjectRoot "msi\build-msi.ps1") @(
                "-Version", $Version,
                "-CudaTag", $cuda,
                "-BuildDir", $buildDir,
                "-StagingDir", $sdkDir,
                "-SkipStaging",
                "-OutputPath", (Join-Path $OutputDir "RiemannEngine-$Version-cu$cuda-x64.msi")
            )
        })
    }
}

# -- Ubuntu: .deb + SDK tarball (Docker BuildKit) ------------------------------

if (-not $SkipDeb) {
    foreach ($cuda in $DebCuda) {
        [void](Invoke-Step "Ubuntu cu$cuda - .deb" {
            Invoke-ChildScript (Join-Path $ProjectRoot "apt\build-apt.ps1") @(
                "-Version", $Version,
                "-CudaVersion", $cuda,
                "-CudaArchitectures", $CudaArchitectures,
                "-Jobs", "$Jobs",
                "-OutputDir", $OutputDir
            )
        })
    }
}

# -- Docker images ---------------------------------------------------------------

if (-not $SkipDocker) {
    foreach ($cuda in $DockerCuda) {
        [void](Invoke-Step "Docker cu$cuda - build$(if (-not $NoPush) { ' + push' })" {
            $versionTag = "$Version-cu$cuda"
            $dockerArgs = @(
                "-CudaVersion", $cuda,
                "-Tag", $versionTag,
                "-User", $DockerUser,
                "-Repo", $DockerRepo,
                "-Jobs", "$Jobs",
                "-CudaArchitectures", $CudaArchitectures
            )
            if (-not $NoPush) { $dockerArgs += "-Push" }
            Invoke-ChildScript (Join-Path $ProjectRoot "docker\docker-build.ps1") $dockerArgs

            $prefix = if ($DockerUser) { "$DockerUser/$DockerRepo" } else { $DockerRepo }
            $floating = "${prefix}:cu$cuda"
            Invoke-Native "docker" @("tag", "${prefix}:$versionTag", $floating)
            if (-not $NoPush) { Invoke-Native "docker" @("push", $floating) }
            if ("$cuda" -eq "134") {
                Invoke-Native "docker" @("tag", "${prefix}:$versionTag", "${prefix}:latest")
                if (-not $NoPush) { Invoke-Native "docker" @("push", "${prefix}:latest") }
            }
        })
    }
}

# -- Checksums + summary ---------------------------------------------------------

if (Test-Path $StageRoot) { Remove-Item -Recurse -Force $StageRoot }

$artifacts = Get-ChildItem -Path $OutputDir -File |
    Where-Object { $_.Extension -in ".msi", ".deb", ".zip", ".gz" } |
    Sort-Object Name
if ($artifacts) {
    $sums = $artifacts | ForEach-Object {
        "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLowerInvariant(), $_.Name
    }
    [System.IO.File]::WriteAllLines((Join-Path $OutputDir "SHA256SUMS.txt"), [string[]]$sums)
}

Write-Banner "Release $Version summary"
$Results | Format-Table Step, Status, Duration, Detail -AutoSize | Out-String -Width 200 | Write-Host
Write-Host "Artifacts in ${OutputDir}:"
$artifacts | ForEach-Object { Write-Host ("  {0,-60} {1,10:N1} MB" -f $_.Name, ($_.Length / 1MB)) }

$failed = @($Results | Where-Object Status -ne "OK")
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "$($failed.Count) step(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "All release steps succeeded." -ForegroundColor Green
exit 0
