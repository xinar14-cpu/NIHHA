# NIHHA Dental Dataset Downloader — interactive bootstrap
# Run via START.bat (double-click) or directly:
#   pwsh -ExecutionPolicy Bypass -File START.ps1

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

Heading "NIHHA Dental Dataset Downloader"
Write-Host "This wizard will:"
Write-Host "  1. Verify Python is installed"
Write-Host "  2. Install/upgrade Python dependencies"
Write-Host "  3. Optionally configure Kaggle and Roboflow API keys"
Write-Host "  4. Download up to ~120 GB into a folder you choose"
Write-Host ""

# ----- 1. Target folder -----
Heading "Step 1 / 5  Target folder"
$target = Ask "Where to put the datasets?" "D:\DATASETS"
try {
    if (-not (Test-Path $target)) { New-Item -ItemType Directory -Force -Path $target | Out-Null }
    Write-Host "  Target ready: $target" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: cannot create $target  -- $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 2
}

# ----- 2. Python check -----
Heading "Step 2 / 5  Python check"
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
    Write-Host "  Python 3.9+ NOT FOUND." -ForegroundColor Red
    Write-Host "  Install from https://python.org/downloads (tick 'Add Python to PATH')."
    Write-Host "  Then re-run START.bat."
    Read-Host "Press Enter to exit"; exit 2
}
Write-Host "  Python OK: $py" -ForegroundColor Green

# ----- 3. Pip dependencies -----
Heading "Step 3 / 5  Python dependencies"
Write-Host "  Installing: requests gdown kaggle roboflow huggingface_hub ..."
& $py -m pip install --quiet --upgrade requests gdown kaggle roboflow huggingface_hub
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARN: pip reported errors. Continuing anyway." -ForegroundColor Yellow
} else {
    Write-Host "  Dependencies OK" -ForegroundColor Green
}

# ----- 4. API keys -----
Heading "Step 4 / 5  API keys (optional but unlock ~50% of the data)"

# 4a. Kaggle
$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleJson = Join-Path $kaggleDir "kaggle.json"
if (Test-Path $kaggleJson) {
    Write-Host "  Kaggle:    token already present at $kaggleJson" -ForegroundColor Green
} else {
    Write-Host "  Kaggle:    no token found. ~8 datasets (~30k photos) will be skipped without it."
    if (YesNo "  Configure Kaggle now?" $true) {
        Write-Host "    Open https://www.kaggle.com/settings -> Create New API Token (downloads kaggle.json)."
        $src = Ask "    Path to your downloaded kaggle.json (blank = skip)"
        if ($src -and (Test-Path $src)) {
            New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null
            Copy-Item -Force -Path $src -Destination $kaggleJson
            Write-Host "    Kaggle token installed." -ForegroundColor Green
        } else {
            Write-Host "    Skipped Kaggle setup (path empty or not found)." -ForegroundColor Yellow
        }
    }
}

# 4b. Roboflow
if ($env:ROBOFLOW_API_KEY) {
    Write-Host "  Roboflow:  ROBOFLOW_API_KEY already set in env" -ForegroundColor Green
} else {
    Write-Host "  Roboflow:  no API key. ~22 datasets (~50k photos) will be skipped without it."
    if (YesNo "  Configure Roboflow now?" $true) {
        Write-Host "    Open https://app.roboflow.com/settings/api -> copy Private API Key."
        $rfkey = Read-Host "    Paste Roboflow Private API Key (blank = skip)"
        if ($rfkey) {
            $env:ROBOFLOW_API_KEY = $rfkey
            Write-Host "    Set for this session." -ForegroundColor Green
            if (YesNo "    Save permanently to your user environment?" $false) {
                [Environment]::SetEnvironmentVariable("ROBOFLOW_API_KEY", $rfkey, "User")
                Write-Host "    Saved permanently (User scope)." -ForegroundColor Green
            }
        } else {
            Write-Host "    Skipped Roboflow setup." -ForegroundColor Yellow
        }
    }
}

# 4c. Git for fallback github clones
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "  Git:       not found. ~4 GitHub datasets without releases will be skipped." -ForegroundColor Yellow
    Write-Host "             Install from https://git-scm.com if you want them." -ForegroundColor Yellow
} else {
    Write-Host "  Git:       OK" -ForegroundColor Green
}

# ----- 5. Confirm and run -----
Heading "Step 5 / 5  Begin download"
Write-Host "  Manifest: 74 datasets queued."
Write-Host "  Target:   $target"
Write-Host "  Resume-safe: re-running the script skips folders that already have data."
Write-Host "  Disk:     up to ~120 GB total (varies by which API keys you set)."
Write-Host ""

$onlyHosts = ""
if (YesNo "  Limit to no-auth hosts only (zenodo + figshare + mendeley + dryad)?  Faster first pass." $false) {
    $onlyHosts = "zenodo,figshare,mendeley,dryad,github,huggingface"
}

if (-not (YesNo "  Start downloading now?" $true)) {
    Write-Host "  Cancelled." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"; exit 0
}

Write-Host ""
$pyArgs = @("download_datasets.py", "--target", $target)
if ($onlyHosts) { $pyArgs += @("--only", $onlyHosts) }

Write-Host "Running: $py $pyArgs" -ForegroundColor Cyan
& $py @pyArgs
$rc = $LASTEXITCODE

# ----- post-run summary -----
Heading "Done"
$logCsv = Join-Path $target "_download_log.csv"
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
} else {
    Write-Host "No log file created (nothing was attempted)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "If any rows are 'requires_auth' or 'manual', re-run START.bat after setting the missing keys."
Write-Host "If 'error', re-run later (network/API hiccup; the script will resume from where it left off)."
Write-Host ""
Read-Host "Press Enter to close"
exit $rc
