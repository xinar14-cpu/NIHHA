#!/usr/bin/env python3
"""
NIHHA dental-dataset auto-downloader.

Reads `downloadable_now.jsonl` and pulls each dataset into
<target>/<dataset_id>/, logging per-record status to
<target>/_download_log.csv.

Strategies by host:
  - zenodo.org             -> Zenodo records API + file URLs
  - figshare.com           -> Figshare articles API + file URLs
  - data.mendeley.com      -> Mendeley public API + file URLs
  - datadryad.org          -> Dryad v2 API
  - github.com             -> latest release assets, else `git clone`
  - drive.google.com       -> gdown (pip install gdown)
  - kaggle.com             -> `kaggle` CLI (needs ~/.kaggle/kaggle.json)
  - universe.roboflow.com  -> roboflow SDK (needs ROBOFLOW_API_KEY env)
  - huggingface.co         -> huggingface_hub.snapshot_download
  - arxiv.org              -> manual flag (no canonical data hosting)
  - everything else        -> manual flag

Usage:
  python download_datasets.py --target "D:\\DATASETS"
  python download_datasets.py --target "/mnt/d/DATASETS" --only zenodo,figshare,mendeley
  python download_datasets.py --target ./out --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr)
    sys.exit(2)


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "downloadable_now.jsonl"

UA = {"User-Agent": "NIHHA-dataset-downloader/1.0 (+research; contact via repo)"}
TIMEOUT = 60


def log(*a):
    print("[downloader]", *a, flush=True)


def safe_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)[:120]


def host_of(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def download_stream(url: str, dest: Path, chunk: int = 1 << 20) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=UA, timeout=TIMEOUT, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as fh:
            for buf in r.iter_content(chunk_size=chunk):
                if not buf:
                    continue
                fh.write(buf)
                total += len(buf)
        return total


# ---------- platform handlers --------------------------------------------------

def handle_zenodo(url: str, outdir: Path) -> tuple[str, str, int]:
    m = re.search(r"zenodo\.org/(?:records|record|doi/[\d./a-z]*?)(\d{6,})", url)
    if not m:
        m = re.search(r"zenodo\.org/.*?(\d{6,})", url)
    if not m:
        return "manual", f"cannot parse zenodo id from {url}", 0
    rid = m.group(1)
    api = f"https://zenodo.org/api/records/{rid}"
    r = requests.get(api, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        return "error", f"zenodo api {r.status_code}", 0
    files = r.json().get("files", [])
    if not files:
        return "empty", "zenodo record has no files (or DOI redirect)", 0
    n = 0
    bytes_total = 0
    for f in files:
        link = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        name = f.get("key") or f.get("filename") or f"file_{n}"
        if not link:
            continue
        try:
            size = download_stream(link, outdir / name)
            bytes_total += size
            n += 1
        except Exception as e:
            log(f"  zenodo file {name} failed: {e}")
    return ("ok" if n else "error"), f"{n} files, {bytes_total/1e6:.1f} MB", n


def handle_figshare(url: str, outdir: Path) -> tuple[str, str, int]:
    m = re.search(r"figshare\.com/articles/(?:dataset/)?[^/]+/(\d{6,})", url)
    if not m:
        m = re.search(r"figshare\.(?:\d+)?\.?(\d{6,})", url)
    if not m:
        m = re.search(r"figshare\.com/.*?(\d{6,})", url)
    if not m:
        return "manual", f"cannot parse figshare id from {url}", 0
    aid = m.group(1)
    api = f"https://api.figshare.com/v2/articles/{aid}/files"
    r = requests.get(api, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        return "error", f"figshare api {r.status_code}", 0
    files = r.json()
    if not files:
        return "empty", "figshare article has no files", 0
    n = 0
    bytes_total = 0
    for f in files:
        link = f.get("download_url")
        name = f.get("name") or f"file_{n}"
        if not link:
            continue
        try:
            size = download_stream(link, outdir / name)
            bytes_total += size
            n += 1
        except Exception as e:
            log(f"  figshare file {name} failed: {e}")
    return ("ok" if n else "error"), f"{n} files, {bytes_total/1e6:.1f} MB", n


def handle_mendeley(url: str, outdir: Path) -> tuple[str, str, int]:
    m = re.search(r"data\.mendeley\.com/datasets/([a-z0-9]+)(?:/(\d+))?", url)
    if not m:
        return "manual", f"cannot parse mendeley slug from {url}", 0
    slug = m.group(1)
    version = m.group(2) or "1"
    api = f"https://data.mendeley.com/public-api/datasets/{slug}/files?folder_id=root&version={version}"
    r = requests.get(api, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        # try without version
        api2 = f"https://data.mendeley.com/public-api/datasets/{slug}/files?folder_id=root"
        r = requests.get(api2, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        return "error", f"mendeley api {r.status_code}", 0
    try:
        files = r.json()
    except Exception:
        return "error", "mendeley api non-json", 0
    if isinstance(files, dict):
        files = files.get("results") or files.get("files") or []
    if not files:
        return "empty", "mendeley returned no files", 0
    n = 0
    bytes_total = 0
    for f in files:
        link = (f.get("content_details") or {}).get("download_url") or f.get("download_url")
        name = f.get("filename") or f.get("name") or f"file_{n}"
        if not link:
            continue
        try:
            size = download_stream(link, outdir / name)
            bytes_total += size
            n += 1
        except Exception as e:
            log(f"  mendeley file {name} failed: {e}")
    return ("ok" if n else "error"), f"{n} files, {bytes_total/1e6:.1f} MB", n


def handle_dryad(url: str, outdir: Path) -> tuple[str, str, int]:
    # Dryad URL pattern: datadryad.org/dataset/doi:10.5061/dryad.<slug>
    m = re.search(r"doi:?10\.\d{4,9}/[^\s/?#]+", url) or re.search(r"10\.\d{4,9}/[^\s/?#]+", url)
    if not m:
        return "manual", f"cannot parse dryad doi from {url}", 0
    doi = m.group(0).replace("doi:", "")
    enc = requests.utils.quote(doi, safe="")
    api = f"https://datadryad.org/api/v2/datasets/doi%3A{enc}/download"
    try:
        size = download_stream(api, outdir / "dryad_bundle.zip")
        return "ok", f"1 zip, {size/1e6:.1f} MB", 1
    except Exception as e:
        return "error", str(e), 0


def handle_github(url: str, outdir: Path) -> tuple[str, str, int]:
    m = re.search(r"github\.com/([^/]+)/([^/?#.]+)", url)
    if not m:
        return "manual", f"cannot parse github repo from {url}", 0
    owner, repo = m.group(1), m.group(2)
    # Try latest release first
    api = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    r = requests.get(api, headers=UA, timeout=TIMEOUT)
    if r.status_code == 200:
        rel = r.json()
        assets = rel.get("assets", [])
        if assets:
            n = 0
            bytes_total = 0
            for a in assets:
                link = a.get("browser_download_url")
                name = a.get("name") or f"asset_{n}"
                if not link:
                    continue
                try:
                    size = download_stream(link, outdir / name)
                    bytes_total += size
                    n += 1
                except Exception as e:
                    log(f"  github asset {name} failed: {e}")
            if n:
                return "ok", f"{n} release assets, {bytes_total/1e6:.1f} MB", n
    # Fallback: git clone
    if shutil.which("git") is None:
        return "manual", "no git, no release assets — clone manually", 0
    target = outdir / repo
    if target.exists():
        return "ok", "git repo already cloned", 1
    cmd = ["git", "clone", "--depth", "1", f"https://github.com/{owner}/{repo}.git", str(target)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode == 0:
        return "ok", "git clone --depth 1", 1
    return "error", f"git clone failed: {res.stderr.strip()[:200]}", 0


def handle_gdrive(url: str, outdir: Path) -> tuple[str, str, int]:
    if shutil.which("gdown") is None:
        try:
            import gdown  # noqa: F401
        except ImportError:
            return "requires_auth", "pip install gdown to enable Google Drive downloads", 0
    # Folder vs file
    m_folder = re.search(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)", url)
    m_file = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)
    try:
        import gdown
        if m_folder:
            gdown.download_folder(id=m_folder.group(1), output=str(outdir), quiet=False, use_cookies=False)
            return "ok", "gdown folder", 1
        if m_file:
            out = outdir / "gdrive_file"
            gdown.download(id=m_file.group(1), output=str(out), quiet=False)
            return "ok", "gdown file", 1
        return "manual", "cannot parse gdrive id", 0
    except Exception as e:
        return "error", f"gdown: {e}", 0


def handle_kaggle(url: str, outdir: Path) -> tuple[str, str, int]:
    if shutil.which("kaggle") is None:
        return "requires_auth", "kaggle CLI not installed (pip install kaggle + ~/.kaggle/kaggle.json)", 0
    m = re.search(r"kaggle\.com/(?:datasets|competitions)/([^/]+/[^/?#]+)", url)
    if not m:
        return "manual", f"cannot parse kaggle slug from {url}", 0
    slug = m.group(1)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(outdir), "--unzip"]
    if "/competitions/" in url:
        comp = slug.split("/")[-1] if "/" in slug else slug
        cmd = ["kaggle", "competitions", "download", "-c", comp, "-p", str(outdir)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode == 0:
        return "ok", "kaggle CLI", 1
    return "error", f"kaggle: {res.stderr.strip()[:200]}", 0


def handle_roboflow(url: str, outdir: Path) -> tuple[str, str, int]:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        return "requires_auth", "set env ROBOFLOW_API_KEY (https://app.roboflow.com/settings/api)", 0
    try:
        from roboflow import Roboflow
    except ImportError:
        return "requires_auth", "pip install roboflow", 0
    m = re.search(r"universe\.roboflow\.com/([^/]+)/([^/?#]+)", url)
    if not m:
        return "manual", f"cannot parse roboflow workspace/project from {url}", 0
    ws, proj = m.group(1), m.group(2)
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(ws).project(proj)
        versions = project.versions()
        if not versions:
            return "error", "no versions for this roboflow project", 0
        latest = versions[-1]
        ds = latest.download("yolov8", location=str(outdir), overwrite=False)
        return "ok", f"roboflow v{latest.version}", 1
    except Exception as e:
        return "error", f"roboflow: {e}", 0


def handle_huggingface(url: str, outdir: Path) -> tuple[str, str, int]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return "requires_auth", "pip install huggingface_hub", 0
    m = re.search(r"huggingface\.co/datasets/([^/]+/[^/?#]+)", url)
    if not m:
        return "manual", f"cannot parse hf dataset slug from {url}", 0
    repo = m.group(1)
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(outdir), local_dir_use_symlinks=False)
        return "ok", "hf snapshot", 1
    except Exception as e:
        return "error", f"hf: {e}", 0


# ---------- dispatch -----------------------------------------------------------

DISPATCH = [
    ("zenodo.org", handle_zenodo),
    ("figshare.com", handle_figshare),
    ("data.mendeley.com", handle_mendeley),
    ("datadryad.org", handle_dryad),
    ("drive.google.com", handle_gdrive),
    ("kaggle.com", handle_kaggle),
    ("universe.roboflow.com", handle_roboflow),
    ("huggingface.co", handle_huggingface),
    ("github.com", handle_github),
]


def pick_handler(url: str):
    h = host_of(url).lower()
    for needle, fn in DISPATCH:
        if needle in h:
            return fn
    return None


# ---------- main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="root folder, e.g. D:\\DATASETS")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="downloadable_now.jsonl path")
    p.add_argument("--only", default="", help="comma list of host filters: zenodo,figshare,mendeley,kaggle,roboflow,github,dryad,gdrive,huggingface")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max", type=int, default=0, help="stop after N successful downloads (0 = no limit)")
    args = p.parse_args()

    target = Path(args.target)
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)

    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}

    records = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log_path = target / "_download_log.csv"
    log_rows = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    log_rows.append(r)
        except Exception:
            pass
    seen_ok = {r["id"] for r in log_rows if r.get("status") == "ok"}

    counts = {"ok": 0, "skipped": 0, "manual": 0, "requires_auth": 0, "error": 0, "empty": 0}
    successes = 0

    new_rows = []
    for rec in records:
        rid = safe_id(rec.get("id", ""))
        url = rec.get("url", "")
        host = host_of(url).lower()
        if only and not any(needle in host for needle in only):
            continue

        outdir = target / rid

        if args.skip_existing and rid in seen_ok:
            log(f"SKIP {rid} (already in log)")
            counts["skipped"] += 1
            continue
        if args.skip_existing and outdir.exists() and any(outdir.iterdir()):
            log(f"SKIP {rid} (folder non-empty)")
            counts["skipped"] += 1
            new_rows.append({"id": rid, "url": url, "status": "ok", "message": "pre-existing", "n_files": "?", "ts": int(time.time())})
            continue

        handler = pick_handler(url)
        if not handler:
            log(f"MANUAL {rid} ({host})")
            counts["manual"] += 1
            new_rows.append({"id": rid, "url": url, "status": "manual", "message": f"no handler for host {host}", "n_files": 0, "ts": int(time.time())})
            continue

        if args.dry_run:
            log(f"DRYRUN {rid} -> {handler.__name__}({url})")
            continue

        log(f"FETCH  {rid} via {handler.__name__}")
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            status, msg, nf = handler(url, outdir)
        except Exception as e:
            status, msg, nf = "error", repr(e)[:200], 0
        counts[status] = counts.get(status, 0) + 1
        new_rows.append({"id": rid, "url": url, "status": status, "message": msg, "n_files": nf, "ts": int(time.time())})
        log(f"  -> {status}: {msg}")

        if status == "ok":
            successes += 1
            if args.max and successes >= args.max:
                log(f"reached --max {args.max}")
                break

    # Write log
    if not args.dry_run:
        all_rows = log_rows + new_rows
        # dedup by id (last wins)
        latest = {}
        for r in all_rows:
            latest[r["id"]] = r
        with open(log_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "url", "status", "message", "n_files", "ts"])
            w.writeheader()
            for r in latest.values():
                w.writerow(r)
        log(f"log -> {log_path}")

    log("=== SUMMARY ===")
    for k, v in counts.items():
        log(f"  {k}: {v}")
    log(f"target: {target}")


if __name__ == "__main__":
    main()
