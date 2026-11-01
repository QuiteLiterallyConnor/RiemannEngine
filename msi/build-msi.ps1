# =============================================================================
# RiemannEngine C++ - MSI Installer Builder
# =============================================================================
#
# Generates a Windows Installer (.msi) package for RiemannEngine C++ using
# WiX Toolset v4+.  The script:
#
#   1. Stages a cmake install into a temporary directory.
#   2. Dynamically generates a WiX v4 source file (.wxs) from the staged tree.
#   3. Converts the LICENSE file to RTF for the install dialog.
#   4. Invokes `wix build` to produce the final MSI.
#
# Prerequisites
# -------------
#   - .NET SDK 6+ (for dotnet tool)
#   - WiX v4:  dotnet tool install --global wix
#   - WiX UI extension:  wix extension add WixToolset.UI.wixext
#   - A successful CMake build of RiemannEngine (produces build/ directory)
#
# Usage
# -----
#   .\build-msi.ps1 -BuildDir ..\build-cu134 -CudaTag 134
#
# The MSI installs to <ProgramFiles>\RiemannEngine\<Version>_<cuXXX|cpu>\.
# Every version/variant combination has its own UpgradeCode, so they install
# side by side; rebuilding the same version/variant replaces the old install.
#
# =============================================================================

[CmdletBinding()]
param(
    # Semantic version embedded in the MSI (Major.Minor.Patch).
    # Defaults to project(VERSION) from CMakeLists.txt.
    [ValidatePattern('^(\d+\.\d+\.\d+)?$')]
    [string]$Version = "",

    # CUDA toolkit tag the build was made with (e.g. 132, 134).
    # Detected from <BuildDir>\CMakeCache.txt when omitted.
    [ValidatePattern('^(\d{3})?$')]
    [string]$CudaTag = "",

    # Path to the CMake build directory (must already be built).
    [string]$BuildDir = "..\build",

    # CMake build type used during the build.
    [ValidateSet("Release", "RelWithDebInfo", "Debug", "MinSizeRel")]
    [string]$BuildType = "Release",

    # Where to stage the cmake install.  A temp directory is used by default.
    [string]$StagingDir = "",

    # Output MSI file path.
    [string]$OutputPath = "",

    # If set, skip the cmake --install staging step and reuse an existing
    # staging directory.  Useful for iterating on the WiX source.
    [switch]$SkipStaging,

    # Mark the build as CPU-only (embedded in the MSI product name).
    [switch]$CpuOnly
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

# -- Paths -----------------------------------------------------------------

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$BuildDir    = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir }
               else { (Resolve-Path (Join-Path $ScriptDir $BuildDir)).Path }

if (-not $StagingDir) {
    $StagingDir = Join-Path $ScriptDir "_staging"
}
if (-not [System.IO.Path]::IsPathRooted($StagingDir)) {
    $StagingDir = Join-Path $ScriptDir $StagingDir
}

if (-not $Version) {
    $cmakeLists = Get-Content (Join-Path $ProjectRoot "CMakeLists.txt") -Raw
    if ($cmakeLists -match 'project\s*\([^)]*VERSION\s+(\d+\.\d+\.\d+)') {
        $Version = $Matches[1]
    } else {
        Write-Host "ERROR: Could not read project VERSION from CMakeLists.txt; pass -Version." -ForegroundColor Red
        exit 1
    }
}

if (-not $CpuOnly -and -not $CudaTag) {
    $cachePath = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-Path $cachePath) {
        $cache = Get-Content $cachePath -Raw
        if ($cache -match '(?m)^CPU_ONLY:BOOL=(ON|TRUE|1)\s*$') {
            $CpuOnly = $true
        } elseif ($cache -match '(?m)^CMAKE_CUDA_COMPILER_VERSION:\w+=(\d+)\.(\d+)') {
            $CudaTag = "$($Matches[1])$($Matches[2])"
        }
    }
    if (-not $CpuOnly -and -not $CudaTag) {
        Write-Host "ERROR: Could not detect the CUDA version of $BuildDir; pass -CudaTag or -CpuOnly." -ForegroundColor Red
        exit 1
    }
}

$Variant = if ($CpuOnly) { "cpu" } else { "cu$CudaTag" }
$InstallSubdir = "${Version}_$Variant"
if (-not $OutputPath) {
    $OutputPath = Join-Path $ScriptDir "RiemannEngine-$Version-$Variant-x64.msi"
}
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $ScriptDir $OutputPath
}

$WxsPath     = Join-Path $ScriptDir "_RiemannEngine.wxs"
$LicenseRtf  = Join-Path $ScriptDir "_License.rtf"
$LicenseSrc  = Join-Path $ProjectRoot "LICENSE"

# Name-based GUIDs keep each version/variant stable across rebuilds while
# letting different versions and CUDA variants coexist.
function New-StableGuid {
    param([string]$Name)
    $md5 = [System.Security.Cryptography.MD5]::Create()
    try {
        $bytes = $md5.ComputeHash([System.Text.Encoding]::UTF8.GetBytes("RiemannEngine-MSI/$Name"))
    } finally {
        $md5.Dispose()
    }
    $bytes[7] = ($bytes[7] -band 0x0F) -bor 0x30
    $bytes[8] = ($bytes[8] -band 0x3F) -bor 0x80
    return ([guid]::new($bytes)).ToString().ToUpperInvariant()
}

$UpgradeCode = New-StableGuid "upgrade/$InstallSubdir"
$RegistryComponentGuid = New-StableGuid "cmake-registry/$InstallSubdir"

# -- Banner ----------------------------------------------------------------

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " RiemannEngine C++ - MSI Installer Builder" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Version      : $Version"
Write-Host "  Variant      : $Variant"
Write-Host "  Install Dir  : <ProgramFiles>\RiemannEngine\$InstallSubdir"
Write-Host "  Build Dir    : $BuildDir"
Write-Host "  Staging Dir  : $StagingDir"
Write-Host "  Output       : $OutputPath"
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# -- Prerequisites ---------------------------------------------------------

if (-not (Get-Command "cmake" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'cmake' not found on PATH." -ForegroundColor Red
    Write-Host "Install CMake 3.18+ and ensure it is on PATH." -ForegroundColor Yellow
    exit 1
}
if (-not (Get-Command "wix" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'wix' not found on PATH." -ForegroundColor Red
    Write-Host "Install WiX v4:  dotnet tool install --global wix" -ForegroundColor Yellow
    exit 1
}

# -- Step 1: Stage the cmake install ----------------------------------------

if (-not $SkipStaging) {
    Write-Host "[1/4] Staging cmake install ..." -ForegroundColor Cyan

    if (-not (Test-Path $BuildDir)) {
        Write-Host "ERROR: Build directory not found: $BuildDir" -ForegroundColor Red
        Write-Host "Build the project first with CMake, then re-run this script." -ForegroundColor Yellow
        exit 1
    }

    $cmakeInstallScript = Join-Path $BuildDir "cmake_install.cmake"
    if (-not (Test-Path $cmakeInstallScript)) {
        Write-Host "ERROR: cmake_install.cmake not found in build directory: $BuildDir" -ForegroundColor Red
        Write-Host "The build directory exists but appears empty or incomplete." -ForegroundColor Yellow
        Write-Host "Run a full CMake configure + build first, e.g.:" -ForegroundColor Yellow
        Write-Host "  cmake -B build -DCMAKE_PREFIX_PATH=<libtorch_path> -G Ninja" -ForegroundColor Yellow
        Write-Host "  cmake --build build --config Release" -ForegroundColor Yellow
        exit 1
    }

    # Wipe previous staging
    if (Test-Path $StagingDir) {
        Remove-Item -Recurse -Force $StagingDir
    }

    & cmake --install $BuildDir --prefix $StagingDir --config $BuildType 2>&1 | Write-Host

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: cmake --install failed." -ForegroundColor Red
        exit 1
    }

    Write-Host "  Staged to: $StagingDir" -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping staging (reusing $StagingDir)" -ForegroundColor DarkGray

    if (-not (Test-Path $StagingDir)) {
        Write-Host "ERROR: Staging directory does not exist: $StagingDir" -ForegroundColor Red
        exit 1
    }
}

# Verify staging dir has content
$stagedFiles = Get-ChildItem -Path $StagingDir -Recurse -File
if ($stagedFiles.Count -eq 0) {
    Write-Host "ERROR: Staging directory is empty. Was the build successful?" -ForegroundColor Red
    exit 1
}
Write-Host "  Found $($stagedFiles.Count) file(s) in staging directory." -ForegroundColor Gray
Write-Host ""

# -- Step 2: Generate License RTF -------------------------------------------

Write-Host "[2/4] Generating license RTF ..." -ForegroundColor Cyan

if (Test-Path $LicenseSrc) {
    $licenseText = Get-Content -Path $LicenseSrc -Raw
    # Escape RTF special characters
    $licenseText = $licenseText.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}')
    # Convert newlines to RTF paragraph breaks
    $licenseText = $licenseText -replace "`r`n", "\par`r`n"
    $licenseText = $licenseText -replace "`n", "\par`r`n"

    $rtfContent = @"
{\rtf1\ansi\deff0
{\fonttbl{\f0\fswiss\fcharset0 Segoe UI;}}
{\colortbl;\red0\green0\blue0;}
\viewkind4\uc1\pard\cf1\f0\fs20
$licenseText
\par
}
"@
    Set-Content -Path $LicenseRtf -Value $rtfContent -Encoding ASCII
    Write-Host "  Created: $LicenseRtf" -ForegroundColor Green
} else {
    Write-Host "  WARNING: LICENSE file not found, skipping license dialog." -ForegroundColor Yellow
    $LicenseRtf = $null
}
Write-Host ""

# -- Step 3: Generate WiX v4 source -----------------------------------------

Write-Host "[3/4] Generating WiX source ..." -ForegroundColor Cyan

# Helpers to produce safe WiX identifiers from paths
function Get-WixSafeId {
    param([string]$Prefix, [string]$RelPath)
    $id = $RelPath -replace '[^a-zA-Z0-9_]', '_'
    $id = $id -replace '_+', '_'
    $id = $id.Trim('_')
    $id = "${Prefix}_${id}"

    # WiX identifiers are limited to 72 characters
    if ($id.Length -gt 72) {
        $hash = [System.BitConverter]::ToString(
            [System.Security.Cryptography.SHA256]::Create().ComputeHash(
                [System.Text.Encoding]::UTF8.GetBytes($RelPath)
            )
        ).Replace("-", "").Substring(0, 8)
        $id = $id.Substring(0, 63) + "_" + $hash
    }
    return $id
}

# Recursively generate XML for a directory subtree.
# Populates $script:componentRefs with component IDs.
$script:componentRefs = [System.Collections.Generic.List[string]]::new()

function Build-DirectoryXml {
    param(
        [string]$PhysicalPath,
        [string]$RelPath,        # relative to staging root (for ID generation)
        [int]$Indent
    )

    $pad = "    " * $Indent
    $xml = [System.Text.StringBuilder]::new()

    # -- Files in this directory --------------------------------------
    $files = Get-ChildItem -Path $PhysicalPath -File -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        $fRel   = if ($RelPath) { "$RelPath/$($f.Name)" } else { $f.Name }
        $compId = Get-WixSafeId "C" $fRel
        $fileId = Get-WixSafeId "F" $fRel

        [void]$xml.AppendLine("$pad<Component Id=`"$compId`" Guid=`"*`">")
        [void]$xml.AppendLine("$pad    <File Id=`"$fileId`" Source=`"$($f.FullName)`" KeyPath=`"yes`" />")
        [void]$xml.AppendLine("$pad</Component>")

        $script:componentRefs.Add($compId)
    }

    # -- Subdirectories --------------------------------------------------
    $dirs = Get-ChildItem -Path $PhysicalPath -Directory -ErrorAction SilentlyContinue
    foreach ($d in $dirs) {
        $dRel  = if ($RelPath) { "$RelPath/$($d.Name)" } else { $d.Name }
        $dirId = Get-WixSafeId "D" $dRel

        [void]$xml.AppendLine("$pad<Directory Id=`"$dirId`" Name=`"$($d.Name)`">")
        $childXml = Build-DirectoryXml -PhysicalPath $d.FullName -RelPath $dRel -Indent ($Indent + 1)
        [void]$xml.Append($childXml)
        [void]$xml.AppendLine("$pad</Directory>")
    }

    return $xml.ToString()
}

# Generate the full directory + component tree from staging
$innerXml = Build-DirectoryXml -PhysicalPath $StagingDir -RelPath "" -Indent 5

# Build ComponentRef list
$compRefXml = ($script:componentRefs | ForEach-Object {
    "            <ComponentRef Id=`"$_`" />"
}) -join "`r`n"

# Register the install in CMake's system package registry so that
# find_package(RiemannEngine) can locate side-by-side installs; the generated
# version file selects the requested version.
$regCompId = "C_CMakePackageRegistry"
$regXml = @"
                <Component Id="$regCompId" Guid="$RegistryComponentGuid">
                    <RegistryValue Root="HKLM"
                                   Key="Software\Kitware\CMake\Packages\RiemannEngine"
                                   Name="$InstallSubdir"
                                   Value="[INSTALLFOLDER]lib\cmake\RiemannEngine"
                                   Type="string"
                                   KeyPath="yes" />
                </Component>
"@
$script:componentRefs.Add($regCompId)
$compRefXml += "`r`n            <ComponentRef Id=`"$regCompId`" />"

# Product display name
$productName = if ($CpuOnly) { "RiemannEngine C++ $Version (CPU)" } else { "RiemannEngine C++ $Version (CUDA $CudaTag)" }

# License variable
$licenseVarXml = ""
if ($LicenseRtf) {
    $licenseVarXml = "        <WixVariable Id=`"WixUILicenseRtf`" Value=`"$LicenseRtf`" />"
}

# -- Assemble final WiX source ----------------------------------------------

$wxsContent = @"
<?xml version="1.0" encoding="UTF-8"?>

<!--
    Auto-generated by build-msi.ps1 - do not edit manually.
    Regenerate with:  .\build-msi.ps1 -Version $Version
-->

<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs"
     xmlns:ui="http://wixtoolset.org/schemas/v4/wxs/ui">

    <Package Name="$productName"
             Version="$Version"
             Manufacturer="RiemannEngine"
             UpgradeCode="$UpgradeCode"
             Compressed="yes"
             InstallerVersion="500">

        <SummaryInformation Description="$productName $Version - C++ library for sparse tensors"
                            Manufacturer="RiemannEngine" />

        <!-- Allow upgrades, block downgrades -->
        <MajorUpgrade AllowSameVersionUpgrades="yes"
                      DowngradeErrorMessage="A newer build of [ProductName] is already installed." />

        <MediaTemplate EmbedCab="yes" />

        <!-- Install UI with directory selection -->
        <ui:WixUI Id="WixUI_InstallDir" InstallDirectory="INSTALLFOLDER" />
$licenseVarXml

        <!-- Feature tree -->
        <Feature Id="Complete"
                 Title="$productName"
                 Description="Installs headers, libraries, and CMake integration files."
                 Level="1"
                 ConfigurableDirectory="INSTALLFOLDER">
$compRefXml
        </Feature>

        <!-- Directory layout -->
        <StandardDirectory Id="ProgramFiles6432Folder">
            <Directory Id="RIEMANNROOT" Name="RiemannEngine">
                <Directory Id="INSTALLFOLDER" Name="$InstallSubdir">
$innerXml
$regXml
                </Directory>
            </Directory>
        </StandardDirectory>

    </Package>
</Wix>
"@

Set-Content -Path $WxsPath -Value $wxsContent -Encoding UTF8
Write-Host "  Generated: $WxsPath" -ForegroundColor Green
Write-Host "  Components: $($script:componentRefs.Count)" -ForegroundColor Gray
Write-Host ""

# -- Step 4: Build the MSI --------------------------------------------------

Write-Host "[4/4] Building MSI with WiX ..." -ForegroundColor Cyan

$wixArgs = @(
    "build"
    "-ext", "WixToolset.UI.wixext"
    "-o", $OutputPath
    $WxsPath
)

Write-Host "  wix $($wixArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

& wix @wixArgs 2>&1 | Write-Host

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: WiX build failed." -ForegroundColor Red
    Write-Host "Check the output above for details." -ForegroundColor Yellow
    exit 1
}

# -- Done -------------------------------------------------------------------

$msiSize = (Get-Item $OutputPath).Length / 1MB
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host " MSI created successfully!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  File : $OutputPath"
Write-Host "  Size : $([math]::Round($msiSize, 2)) MB"
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# -- Cleanup intermediate files ----------------------------------------------

# Leave staging dir for debugging; remove generated WiX source and RTF
# Uncomment the following lines to auto-clean:
# Remove-Item -Path $WxsPath -ErrorAction SilentlyContinue
# Remove-Item -Path $LicenseRtf -ErrorAction SilentlyContinue
