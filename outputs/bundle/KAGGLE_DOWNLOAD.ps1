# NIHHA Kaggle-only downloader (Windows wizard)
# Run via KAGGLE_DOWNLOAD.bat (double-click) or directly.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

function Heading([string]$t) {
    Write-Host ""
    Write-Host ("=" * 64) -ForegroundColor Cyan
    Write-Host $t -ForegroundColor Cyan
    Write-Host ("=" * 64) -ForegroundColor Cyan
}

function Ask([string]$prompt, [string]$default = "") {
    if ($default) {
        $r = Read-Host "$prompt [$default]"
        if ([string]::IsNullOrWhiteSpace($r)) { return $default }
        return $r
    }
    return (Read-Host $prompt)
}

function YesNo([string]$prompt, [bool]$default = $true) {
    $hint = if ($default) { "(Y/n)" } else { "(y/N)" }
    $r = Read-Host "$prompt $hint"
    if ([string]::IsNullOrWhiteSpace($r)) { return $default }
    return ($r -match '^[yY]')
}

Heading "NIHHA Kaggle-only Dataset Downloader"
Write-Host "Pulls 8 Kaggle dental datasets:"
Write-Host "  - salmansajid05/oral-diseases               (~12,653 photos)"
Write-Host "  - maazmakhdoom/dental-cavity-detection      (~500 photos)"
Write-Host "  - zaidpy/oral-cancer-dataset                (~131 photos)"
Write-Host "  - bavithravairam/mouth-ulcer                (?)"
Write-Host "  - bavithravairam/oral-ulcer                 (?)"
Write-Host "  - shlokmohanty/ulcer-classification         (?)"
Write-Host "  - santhoshsivang/calculus-dataset           (?)"
Write-Host "  - muhammadatef/oral-cancer-images           (?)"
Write-Host ""

# 1. Target
Heading "Step 1 / 4  Target folder"
$target = Ask "Where to put the Kaggle datasets?" "D:\DATASETS"
if (-not (Test-Path $target)) { New-Item -ItemType Directory -Force -Path $target | Out-Null }
Write-Host "  Target: $target" -ForegroundColor Green

# 2. Python
Heading "Step 2 / 4  Python check"
$py = $null
foreach ($c in @("python", "py", "python3")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) {
        try {
            $ver = & $c -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver -match '^3\.(9|1[0-9])') { $py = $c; break }
        } catch {}
    }
}
if (-not $py) {
    Write-Host "  Python 3.9+ NOT FOUND. Install from https://python.org/downloads" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 2
}
Write-Host "  Python OK: $py" -ForegroundColor Green

# 3. Kaggle CLI + token
Heading "Step 3 / 4  Kaggle CLI and API token"
& $py -m pip install --quiet --upgrade kaggle
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARN: pip install kaggle had errors. Continuing." -ForegroundColor Yellow
} else {
    Write-Host "  Kaggle CLI installed" -ForegroundColor Green
}

$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleJson = Join-Path $kaggleDir "kaggle.json"
if (Test-Path $kaggleJson) {
    Write-Host "  Token already at $kaggleJson" -ForegroundColor Green
} else {
    Write-Host "  No token found at $kaggleJson" -ForegroundColor Yellow
    Write-Host "  1) Open https://www.kaggle.com/settings -> Create New API Token"
    Write-Host "     This downloads kaggle.json into your Downloads folder."
    Write-Host "  2) Paste full path to kaggle.json below."
    $src = Ask "  Path to kaggle.json"
    if ($src -and (Test-Path $src)) {
        New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null
        Copy-Item -Force -Path $src -Destination $kaggleJson
        Write-Host "  Token installed at $kaggleJson" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: kaggle.json not provided. Cannot continue." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 2
    }
}

# 4. Run
Heading "Step 4 / 4  Begin download"
if (-not (YesNo "  Start downloading 8 Kaggle datasets to $target now?" $true)) {
    Write-Host "  Cancelled." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"; exit 0
}

$script = Join-Path $here "kaggle_only.py"
Write-Host ""
Write-Host "Running: $py $script --target $target" -ForegroundColor Cyan
& $py $script --target $target
$rc = $LASTEXITCODE

Heading "Done"
$logCsv = Join-Path $target "_kaggle_log.csv"
if (Test-Path $logCsv) {
    Write-Host "Log: $logCsv"
    try {
        $rows = Import-Csv $logCsv
        $byStatus = $rows | Group-Object status | Sort-Object Count -Descending
        Write-Host ""
        Write-Host "Status counts:"
        foreach ($g in $byStatus) {
            Write-Host ("  {0,-15} {1,5}" -f $g.Name, $g.Count)
        }
    } catch {}
}
Write-Host ""
Read-Host "Press Enter to close"
exit $rc
