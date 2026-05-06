#!/usr/bin/env python3
"""
Kaggle-only NIHHA dataset downloader.

Pulls the 8 Kaggle datasets from downloadable_now.jsonl into
<target>/<dataset_id>/ using the official Kaggle CLI.

Requires:
  pip install kaggle
  Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
  or ~/.kaggle/kaggle.json (Mac/Linux). Get yours at
  https://www.kaggle.com/settings -> Create New API Token.

Usage:
  python kaggle_only.py --target "D:\\DATASETS"
  python kaggle_only.py --target "D:\\DATASETS" --dry-run
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

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "downloadable_now.jsonl"


def log(*a):
    print("[kaggle]", *a, flush=True)


def have_kaggle_cli() -> bool:
    return shutil.which("kaggle") is not None


def have_kaggle_token() -> Path | None:
    home = Path.home()
    candidates = [home / ".kaggle" / "kaggle.json"]
    if os.name == "nt" and "USERPROFILE" in os.environ:
        candidates.append(Path(os.environ["USERPROFILE"]) / ".kaggle" / "kaggle.json")
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_slug(url: str):
    m = re.search(r"kaggle\.com/datasets/([^/]+/[^/?#]+)", url)
    if m:
        return ("dataset", m.group(1))
    m = re.search(r"kaggle\.com/competitions/([^/?#]+)", url)
    if m:
        return ("competition", m.group(1))
    return (None, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-download even if folder is non-empty")
    args = p.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    if not have_kaggle_cli():
        log("ERROR: kaggle CLI not found. Run: pip install kaggle")
        sys.exit(2)
    token = have_kaggle_token()
    if not token:
        log("ERROR: kaggle.json token not found.")
        log("  1) Visit https://www.kaggle.com/settings -> Create New API Token")
        log("  2) Move kaggle.json into:")
        if os.name == "nt":
            log(f"     {os.environ.get('USERPROFILE','%USERPROFILE%')}\\.kaggle\\kaggle.json")
        else:
            log(f"     {Path.home() / '.kaggle' / 'kaggle.json'}")
        sys.exit(2)
    log(f"Kaggle token: {token}")

    records = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    kaggle_recs = [r for r in records if "kaggle.com" in (r.get("url") or "").lower()]
    log(f"Kaggle datasets in manifest: {len(kaggle_recs)}")

    log_path = target / "_kaggle_log.csv"
    rows = []
    counts = {"ok": 0, "skipped": 0, "error": 0, "manual": 0}

    for rec in kaggle_recs:
        rid = re.sub(r"[^a-zA-Z0-9_.-]+", "_", rec.get("id", ""))[:120]
        url = rec.get("url", "")
        kind, slug = parse_slug(url)
        outdir = target / rid

        if not slug:
            log(f"MANUAL {rid} ({url})")
            counts["manual"] += 1
            rows.append({"id": rid, "slug": "", "status": "manual", "message": "cannot parse Kaggle slug", "url": url})
            continue

        if not args.force and outdir.exists() and any(outdir.iterdir()):
            log(f"SKIP   {rid} (folder non-empty)")
            counts["skipped"] += 1
            rows.append({"id": rid, "slug": slug, "status": "skipped", "message": "pre-existing", "url": url})
            continue

        if args.dry_run:
            log(f"DRYRUN {rid} -> kaggle {kind} download {slug}")
            continue

        outdir.mkdir(parents=True, exist_ok=True)
        if kind == "dataset":
            cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(outdir), "--unzip"]
        else:
            cmd = ["kaggle", "competitions", "download", "-c", slug, "-p", str(outdir)]

        log(f"FETCH  {rid} -> {' '.join(cmd)}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            counts["error"] += 1
            rows.append({"id": rid, "slug": slug, "status": "error", "message": "timeout 30min", "url": url})
            log("  -> timeout 30 min")
            continue

        if r.returncode == 0:
            counts["ok"] += 1
            rows.append({"id": rid, "slug": slug, "status": "ok", "message": "downloaded", "url": url})
            log(f"  -> ok")
        else:
            counts["error"] += 1
            err = (r.stderr or r.stdout or "").strip()[:300]
            rows.append({"id": rid, "slug": slug, "status": "error", "message": err, "url": url})
            log(f"  -> error: {err}")

    if not args.dry_run:
        with open(log_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "slug", "status", "message", "url"])
            w.writeheader()
            for row in rows:
                w.writerow(row)
        log(f"Log: {log_path}")

    log("=== summary ===")
    for k, v in counts.items():
        log(f"  {k}: {v}")
    log(f"target: {target}")


if __name__ == "__main__":
    main()
