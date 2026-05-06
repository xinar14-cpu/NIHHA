# NIHHA Dental Dataset Auto-Downloader — Instructions

What it does: walks the 74-record `downloadable_now.jsonl` and pulls each dataset into `<target>\<dataset_id>\` on your Windows D: drive (or any other folder). Skips already-downloaded folders. Logs every attempt to `<target>\_download_log.csv`.

Coverage estimate: ~30–35 datasets pull cleanly with no auth (Zenodo, Figshare, Mendeley, Dryad, GitHub releases, Hugging Face). Another ~30 require API keys (Kaggle 8, Roboflow 22). The rest (~10) require Google Drive access or manual visit and are flagged in the log.

## Quick start

```powershell
# 1. From a fresh PowerShell window in the cloned repo root:
git pull origin claude/dental-dataset-search-3Vwtp

# 2. Run the downloader (will prompt for installs as needed):
pwsh -ExecutionPolicy Bypass -File outputs\DOWNLOAD.ps1 -Target "D:\DATASETS"
```

That's it. The script:
1. Verifies Python 3.9+ is installed.
2. Installs/updates pip deps: `requests gdown kaggle roboflow huggingface_hub`.
3. Probes for Kaggle/Roboflow credentials and warns (does not abort) if missing.
4. Hands off to `outputs\download_datasets.py` with your target.

## Adding API keys (optional but recommended)

### Kaggle (~8 datasets, ~30k photos including salmansajid05 12,653 imgs)

1. Visit https://www.kaggle.com/settings → "Create New API Token". Downloads `kaggle.json`.
2. Move it to `%USERPROFILE%\.kaggle\kaggle.json`. Create the folder if missing:
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
   Move-Item -Force "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\"
   ```
3. Re-run the downloader. Kaggle entries will succeed.

### Roboflow Universe (~22 datasets, ~50k photos)

1. Visit https://app.roboflow.com/settings/api → copy your Private API Key.
2. Set it for the current PowerShell session only:
   ```powershell
   $env:ROBOFLOW_API_KEY = "rf_your_key_here"
   ```
   Or permanently: System Properties → Environment Variables → New User Variable
   `ROBOFLOW_API_KEY = rf_your_key_here`. Restart PowerShell.
3. Re-run the downloader.

## Useful flags

```powershell
# Filter to specific platforms (comma list, partial host match):
pwsh outputs\DOWNLOAD.ps1 -Target "D:\DATASETS" -Only "zenodo,figshare,mendeley"

# Dry run (lists what would be fetched, no downloads):
pwsh outputs\DOWNLOAD.ps1 -Target "D:\DATASETS" -DryRun

# Stop after N successful downloads:
pwsh outputs\DOWNLOAD.ps1 -Target "D:\DATASETS" -Max 5

# Resume — running it again will SKIP folders that already have content,
# so safe to interrupt with Ctrl+C and restart.
```

## Log format

`D:\DATASETS\_download_log.csv` — one row per dataset, columns:

| column | meaning |
|---|---|
| `id` | dataset_id (matches folder under `D:\DATASETS\`) |
| `url` | source URL |
| `status` | `ok` / `skipped` / `manual` / `requires_auth` / `empty` / `error` |
| `message` | short reason or file-count summary |
| `n_files` | number of files downloaded (best-effort) |
| `ts` | unix timestamp |

Open in Excel and filter by `status` column to see what failed and why.

## Status meanings

| status | what to do |
|---|---|
| `ok` | done, files in folder |
| `skipped` | folder already had content; rerun with `-NoSkipExisting` (Python flag) to force |
| `manual` | host not auto-handled — visit URL and download by hand |
| `requires_auth` | install Kaggle CLI / set ROBOFLOW_API_KEY / install gdown — see above |
| `empty` | API returned no files (rare; usually metadata-only Zenodo records) |
| `error` | network failure or API error; rerun later |

## What gets skipped automatically (and why)

- **MeMoSA / BDJ Cairo / OII-DS / Sun Yat-sen / Bengbu / Charité / AIHub Korea / CLASEG / BSPC / MIH-3241** etc. — these are in `requires_request.csv`, not in `downloadable_now.jsonl`. The downloader does **not** touch them. See `outputs\REPORT.md` and `outputs\requires_request.csv` for the outreach workflow.
- **Roboflow Universe entries with stale slugs** — some Universe URLs in our manifest are search queries, not project pages. They will surface as `error` or `manual`. Patch the URL in `downloadable_now.jsonl` and rerun if you spot them.
- **Google Drive** — `gdown` is installed automatically. If a folder is restricted to "anyone with the link," it works. If "request access," it fails — open the URL in a browser and request manually.

## Troubleshooting

**`Python not found`** — install from https://python.org (3.10+). Tick "Add to PATH" during install.

**Kaggle 401 / "no API token"** — verify `%USERPROFILE%\.kaggle\kaggle.json` exists and chmod 600 isn't needed on Windows; restart PowerShell after moving the file.

**Roboflow `Project does not exist`** — the URL slug is stale. Visit `universe.roboflow.com`, search the project name, copy the corrected `<workspace>/<project>` from the new URL into `downloadable_now.jsonl`, rerun.

**`SSLError` / certificate issues** — corporate VPN intercepting; try from home network or set `REQUESTS_CA_BUNDLE` to your corporate CA cert.

**Out of disk space** — the full pull is ~30–60 GB. Use `-Only "zenodo,figshare,mendeley"` for a 5–10 GB subset first.

## Resuming after interrupt

Re-run the same command. The log is read first; any `id` whose folder is non-empty (or whose log row is `ok`) is skipped. To force re-fetch, delete the folder.

## Manual fallback for `requires_auth` / `manual` rows

Open `D:\DATASETS\_download_log.csv` in Excel, filter `status` = `requires_auth` or `manual`, copy the `url` column, paste into a browser, download manually into `D:\DATASETS\<id>\`. The log will mark them `ok` on next dry-run scan.

## After the download

```powershell
# Quick sanity check: count files per dataset
Get-ChildItem D:\DATASETS -Directory | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -Recurse -File | Measure-Object).Count
    "{0,-60} {1,8}" -f $_.Name, $n
} | Sort-Object
```

A successful run typically produces ~30k–80k photos across 30–50 dataset folders, depending on which auth keys are configured.
