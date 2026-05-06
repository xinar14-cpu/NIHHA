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
    Write-Host "  Kaggle CLI installed/updated" -ForegroundColor Green
}

$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleJson = Join-Path $kaggleDir "kaggle.json"
$accessTokenFile = Join-Path $kaggleDir "access_token"

# Detect any existing token (either old kaggle.json or new access_token or env var)
$haveToken = $false
if (Test-Path $kaggleJson)        { Write-Host "  Found classic kaggle.json at $kaggleJson"      -ForegroundColor Green; $haveToken = $true }
if (Test-Path $accessTokenFile)   { Write-Host "  Found access_token at $accessTokenFile"        -ForegroundColor Green; $haveToken = $true }
if ($env:KAGGLE_API_TOKEN)        { Write-Host "  KAGGLE_API_TOKEN env var already set"          -ForegroundColor Green; $haveToken = $true }

if (-not $haveToken) {
    Write-Host "  No Kaggle credentials found." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Open https://www.kaggle.com/settings -> 'Create New API Token'."
    Write-Host "  Kaggle now issues two formats. Either is fine:"
    Write-Host ""
    Write-Host "    A) NEW Personal Access Token starting with 'KGAT_...'"
    Write-Host "       Just copy the token string (icon next to the field)."
    Write-Host ""
    Write-Host "    B) CLASSIC kaggle.json (older accounts)"
    Write-Host "       The browser downloads a kaggle.json file."
    Write-Host ""
    $choice = Ask "  Which one do you have? Type A or B" "A"
    $choice = $choice.Trim().ToUpper()

    if ($choice -eq "A") {
        $tok = Read-Host "  Paste the KGAT_... token (input is visible)"
        $tok = $tok.Trim()
        if (-not $tok) {
            Write-Host "  ERROR: empty token. Cannot continue." -ForegroundColor Red
            Read-Host "Press Enter to exit"; exit 2
        }
        New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null
        # Write WITHOUT trailing newline (Kaggle CLI is strict about this on some versions)
        [System.IO.File]::WriteAllText($accessTokenFile, $tok)
        # Set env var for current session as a belt-and-braces fallback
        $env:KAGGLE_API_TOKEN = $tok
        Write-Host "  Token saved to $accessTokenFile" -ForegroundColor Green
        Write-Host "  Also set KAGGLE_API_TOKEN env var for this session" -ForegroundColor Green
        if (YesNo "  Save KAGGLE_API_TOKEN permanently to your User environment too?" $false) {
            [Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", $tok, "User")
            Write-Host "  Saved to User env (takes effect in new shells)." -ForegroundColor Green
        }
    }
    elseif ($choice -eq "B") {
        $src = Ask "  Path to kaggle.json (e.g. C:\Users\You\Downloads\kaggle.json)"
        if ($src -and (Test-Path $src)) {
            New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null
            Copy-Item -Force -Path $src -Destination $kaggleJson
            Write-Host "  kaggle.json installed at $kaggleJson" -ForegroundColor Green
        } else {
            Write-Host "  ERROR: kaggle.json not provided / not found. Cannot continue." -ForegroundColor Red
            Read-Host "Press Enter to exit"; exit 2
        }
    }
    else {
        Write-Host "  ERROR: pick A or B." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 2
    }
}

# Smoke test: ask Kaggle CLI to do something cheap
Write-Host ""
Write-Host "  Testing Kaggle CLI auth..." -ForegroundColor Cyan
$probe = & kaggle datasets list --max-size 1 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARN: Kaggle auth probe failed. The token may be invalid or your kaggle CLI is too old." -ForegroundColor Yellow
    Write-Host "  Output: $probe" -ForegroundColor Yellow
    if (-not (YesNo "  Continue anyway?" $false)) {
        Read-Host "Press Enter to exit"; exit 2
    }
} else {
    Write-Host "  Kaggle auth OK" -ForegroundColor Green
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
