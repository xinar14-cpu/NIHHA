NIHHA Dental Dataset Downloader
================================

QUICK START
-----------
1. Unzip this archive somewhere (e.g. C:\NIHHA-Downloader\).
2. Double-click START.bat.
3. Follow the on-screen wizard. It will:
   - Verify Python 3.9+ is installed.
   - Install Python dependencies.
   - Optionally configure Kaggle and Roboflow API keys.
   - Download up to 120 GB into the folder you choose
     (default: D:\DATASETS).
4. After it finishes, open D:\DATASETS\_download_log.csv
   in Excel to see the per-dataset status.


PREREQUISITES
-------------
* Windows 10/11.
* Python 3.9 or newer, installed with "Add Python to PATH".
  Get it at https://python.org/downloads if missing.
* Optional: Git (for GitHub datasets without release assets).


API KEYS (optional, but unlock ~50% of the data)
------------------------------------------------
The wizard offers to set these for you. Get them in advance to save time.

* Kaggle (~8 datasets, ~30k photos)
  - Visit https://www.kaggle.com/settings
  - Click "Create New API Token" -> downloads kaggle.json
  - Wizard will copy it into %USERPROFILE%\.kaggle\

* Roboflow Universe (~22 datasets, ~50k photos)
  - Visit https://app.roboflow.com/settings/api
  - Copy your "Private API Key"
  - Wizard will store it as the ROBOFLOW_API_KEY env variable
    (ask "Save permanently?" Yes if you want it for next time)


DISK SPACE
----------
* No-auth hosts only (Zenodo + Figshare + Mendeley + Dryad + GitHub + HF):
  approx 5 to 25 GB across ~30 datasets. Wizard offers this as a fast
  first pass.
* Full pull with both API keys: 60 to 120 GB across ~50 datasets.

You can interrupt with Ctrl+C and resume by running START.bat again
(folders that already contain data are skipped).


WHAT IS NOT DOWNLOADED
----------------------
Datasets behind email/DUA gates (MeMoSA 30k, BDJ Cairo 9k, OII-DS 15k,
Sun Yat-sen 7.6k, Bengbu 7.2k, Charite 5.3k, AIHub Korea, CLASEG, BSPC,
MIH-3241, etc.) are NOT in this bundle. They require manual outreach
to the corresponding authors. See requires_request.csv in the main
project repo.


TROUBLESHOOTING
---------------
* "Python not found"
  Install from python.org and tick "Add Python to PATH" during install.
  Restart the terminal/window.

* Kaggle says 401 Unauthorized
  Verify %USERPROFILE%\.kaggle\kaggle.json exists and contains your
  credentials in JSON form. Restart START.bat.

* Roboflow "Project does not exist"
  The URL slug in our manifest may be stale. Open downloadable_now.jsonl
  in a text editor, fix the universe.roboflow.com URL, save, re-run.

* SSLError / certificate issues
  Likely a corporate network / VPN intercepting HTTPS. Try from a home
  network, or set REQUESTS_CA_BUNDLE to your corporate root CA.

* Out of disk space
  Pick a different drive in Step 1, or limit to no-auth hosts only
  in Step 5. You can also delete folders you do not need after the run.


FILES IN THIS BUNDLE
--------------------
START.bat              Double-click entry point (Windows)
START.ps1              The interactive PowerShell wizard
download_datasets.py   The actual per-host downloader
downloadable_now.jsonl 74-record manifest of open datasets
README.txt             This file
