# NIHHA dental-dataset auto-downloader (Windows wrapper)
# Usage:
#   pwsh -ExecutionPolicy Bypass -File outputs\DOWNLOAD.ps1 -Target "D:\DATASETS"
#   pwsh outputs\DOWNLOAD.ps1 -Target "D:\DATASETS" -Only "zenodo,figshare,mendeley"
#   pwsh outputs\DOWNLOAD.ps1 -Target "D:\DATASETS" -DryRun

param(
    [Parameter(Mandatory=$true)] [string]$Target,
    [string]$Only = "",
    [switch]$DryRun,
    [int]$Max = 0
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Need([string]$cmd, [string]$hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "MISSING: $cmd  ($hint)" -ForegroundColor Yellow
        return $false
    }
    return $true
}

# 1. Python check
$py = $null
foreach ($c in @("python", "py", "python3")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break }
}
if (-not $py) {
    Write-Host "ERROR: Python 3.9+ not found. Install from python.org." -ForegroundColor Red
    exit 2
}
Write-Host "Python: $py"

# 2. Pip dependencies (quiet install)
Write-Host "Installing/updating Python deps..." -ForegroundColor Cyan
& $py -m pip install --quiet --upgrade requests gdown kaggle roboflow huggingface_hub
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARN: pip install reported errors; continuing anyway." -ForegroundColor Yellow
}

# 3. Optional auth probes
$kaggleJson = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleJson)) {
    Write-Host "NOTE: Kaggle API token not found at $kaggleJson" -ForegroundColor Yellow
    Write-Host "      ~8 Kaggle datasets will be skipped. To enable:" -ForegroundColor Yellow
    Write-Host "      1) https://www.kaggle.com/settings -> Create API Token" -ForegroundColor Yellow
    Write-Host "      2) Move kaggle.json to $kaggleJson" -ForegroundColor Yellow
}

if (-not $env:ROBOFLOW_API_KEY) {
    Write-Host "NOTE: ROBOFLOW_API_KEY env var not set." -ForegroundColor Yellow
    Write-Host "      ~22 Roboflow datasets will be skipped. To enable:" -ForegroundColor Yellow
    Write-Host "      1) https://app.roboflow.com/settings/api -> copy Private API Key" -ForegroundColor Yellow
    Write-Host "      2) `$env:ROBOFLOW_API_KEY = 'rf_xxx'  (this session only)" -ForegroundColor Yellow
    Write-Host "         OR set it permanently via System Properties > Environment Variables" -ForegroundColor Yellow
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "NOTE: git not found. ~4 GitHub datasets without releases will be skipped." -ForegroundColor Yellow
}

# 4. Dispatch
$argsList = @("--target", $Target)
if ($Only)   { $argsList += @("--only", $Only) }
if ($DryRun) { $argsList += "--dry-run" }
if ($Max)    { $argsList += @("--max", "$Max") }

$script = Join-Path $here "download_datasets.py"
Write-Host "Running: $py $script $argsList" -ForegroundColor Cyan
& $py $script @argsList
$code = $LASTEXITCODE

Write-Host ""
Write-Host "Done. Log: $Target\_download_log.csv" -ForegroundColor Green
exit $code
