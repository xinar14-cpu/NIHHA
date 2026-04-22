#!/usr/bin/env python3
"""Dedupe + summarize iter_1 JSONLs. Produces summary.jsonl, master_manifest.jsonl, REPORT.md, updated config.json."""
import json, os, re, sys
from collections import defaultdict

ITER = 1
BASE = "/home/user/NIHHA/.claude/dental-scan"
ITER_DIR = f"{BASE}/iter_{ITER}"
AGENT_FILES = sorted(f for f in os.listdir(ITER_DIR) if f.startswith("agent_") and f.endswith(".jsonl"))

def canonical_url(u):
    if not u:
        return ""
    u = re.sub(r"[?#].*$", "", u).rstrip("/").lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u

def load_all():
    records = []
    for f in AGENT_FILES:
        path = os.path.join(ITER_DIR, f)
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["_source_file"] = f
                    records.append(rec)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {f}: bad JSON line: {e}", file=sys.stderr)
    return records

def dedupe_key(rec):
    doi = (rec.get("doi") or "").strip().lower()
    if doi:
        return ("doi", doi)
    url = canonical_url(rec.get("url") or "")
    if url:
        return ("url", url)
    title = (rec.get("title") or "").strip().lower()
    n = rec.get("num_images") or 0
    return ("title_n", title, n)

def merge(existing, new):
    merged = dict(existing)
    for k, v in new.items():
        if k == "_source_file":
            merged.setdefault("_source_files", [existing.get("_source_file")])
            if v not in merged["_source_files"]:
                merged["_source_files"].append(v)
            continue
        if v in (None, "", [], {}) or (existing.get(k) not in (None, "", [], {})):
            continue
        merged[k] = v
    agents = set()
    for s in existing.get("_source_files", [existing.get("_source_file")]) + [new.get("_source_file")]:
        if s:
            agents.add(s)
    merged["_source_files"] = sorted(agents)
    merged["_agent_origins"] = sorted(set(
        (existing.get("agent_origin"),) if isinstance(existing.get("agent_origin"), str) else tuple(existing.get("_agent_origins", []))
    ) | {new.get("agent_origin")})
    merged["_agent_origins"] = [x for x in merged["_agent_origins"] if x]
    return merged

def dedupe(records):
    by_key = {}
    for rec in records:
        k = dedupe_key(rec)
        if k in by_key:
            by_key[k] = merge(by_key[k], rec)
        else:
            rec = dict(rec)
            rec["_agent_origins"] = [rec.get("agent_origin")] if rec.get("agent_origin") else []
            rec["_source_files"] = [rec.get("_source_file")]
            by_key[k] = rec
    return list(by_key.values())

def metrics(unique):
    by_region = defaultdict(int)
    by_license = defaultdict(int)
    by_agent = defaultdict(int)
    by_modality_mixed = defaultdict(int)
    by_requires_request = defaultdict(int)
    views_hits = defaultdict(int)
    classes_hits = defaultdict(int)
    total_images = 0
    for r in unique:
        by_region[r.get("region_origin") or "unknown"] += 1
        by_license[r.get("license") or "unknown"] += 1
        for a in r.get("_agent_origins", []):
            by_agent[a] += 1
        by_modality_mixed[bool(r.get("modality_mixed"))] += 1
        by_requires_request[bool(r.get("requires_request"))] += 1
        for v in (r.get("views_covered") or []):
            views_hits[v] += 1
        for c in (r.get("classes") or []):
            classes_hits[c] += 1
        ni = r.get("num_images")
        if isinstance(ni, int):
            total_images += ni
    return {
        "total_unique": len(unique),
        "total_images": total_images,
        "by_region": dict(by_region),
        "by_license": dict(by_license),
        "by_agent_origin": dict(by_agent),
        "by_modality_mixed": {str(k): v for k, v in by_modality_mixed.items()},
        "by_requires_request": {str(k): v for k, v in by_requires_request.items()},
        "views_hits": dict(views_hits),
        "classes_hits": dict(classes_hits),
    }

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    raw = load_all()
    unique = dedupe(raw)
    m = metrics(unique)

    summary_path = f"{ITER_DIR}/summary.jsonl"
    write_jsonl(summary_path, unique)

    master_path = f"{BASE}/master_manifest.jsonl"
    master_ids = set()
    if os.path.exists(master_path):
        with open(master_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    master_ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    pass
    new_unique = [r for r in unique if r.get("id") not in master_ids]
    with open(master_path, "a", encoding="utf-8") as f:
        for r in new_unique:
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Update config.json
    config_path = f"{BASE}/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    found_ids = [r.get("id") for r in unique if r.get("id")]
    cfg["exclude_ids"] = sorted(set(cfg.get("exclude_ids", []) + found_ids))
    cfg["iteration"] = ITER + 1
    cfg["stop_recommended"] = len(new_unique) < cfg.get("stop_threshold_new_unique", 5)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # REPORT.md
    report = []
    report.append(f"# Iteration {ITER} — Dental Dataset Scan REPORT\n")
    report.append(f"- Input files: {len(AGENT_FILES)}")
    report.append(f"- Raw records: **{len(raw)}**")
    report.append(f"- Unique (deduped): **{m['total_unique']}**")
    report.append(f"- New vs master_manifest: **{len(new_unique)}**")
    report.append(f"- Total images across unique datasets: **{m['total_images']:,}**")
    report.append(f"- stop_recommended (next iter): **{cfg['stop_recommended']}**\n")

    report.append("## Coverage by agent origin")
    report.append("| Agent | Unique records contributed |")
    report.append("|---|---|")
    for k, v in sorted(m["by_agent_origin"].items(), key=lambda x: -x[1]):
        report.append(f"| {k} | {v} |")

    report.append("\n## Coverage by region")
    report.append("| Region | Count |")
    report.append("|---|---|")
    for k, v in sorted(m["by_region"].items(), key=lambda x: -x[1]):
        report.append(f"| {k} | {v} |")

    report.append("\n## License distribution")
    report.append("| License | Count |")
    report.append("|---|---|")
    for k, v in sorted(m["by_license"].items(), key=lambda x: -x[1]):
        report.append(f"| {k} | {v} |")

    report.append("\n## Views coverage")
    report.append("| View | Datasets hitting view |")
    report.append("|---|---|")
    for k, v in sorted(m["views_hits"].items(), key=lambda x: -x[1]):
        report.append(f"| {k} | {v} |")

    report.append("\n## Requires-request vs open-download")
    report.append(f"- Open download: {m['by_requires_request'].get('False', 0)}")
    report.append(f"- Requires request: {m['by_requires_request'].get('True', 0)}")

    report.append("\n## Gap analysis")
    gaps = []
    if m["by_agent_origin"].get("disease-noncarious", 0) < 3:
        gaps.append("- Non-carious zone thin: MIH/fluorosis datasets mostly request-only. Iter 2: author-contact batch.")
    if m["by_agent_origin"].get("restorative", 0) < 3:
        gaps.append("- Restorative zone under-covered: drill Roboflow per-project; contact Osaka prosthesis authors.")
    if not any("SDF" in (r.get("notes") or "") or "silver diamine" in (r.get("notes") or "").lower() for r in unique):
        gaps.append("- No SDF-treated pediatric photos found — search iter 2 with explicit AAPD/medRxiv queries.")
    if m["views_hits"].get("upper_occlusal", 0) < 5:
        gaps.append("- Upper occlusal view under-represented — target multi-view collections next iter.")
    ru_count = sum(1 for r in unique if (r.get("region_origin") or "").lower() in {"russia", "belarus", "kazakhstan", "ukraine"})
    if ru_count < 3:
        gaps.append("- RU/CIS representation low — iter 2: probe hub.sfedu.ru, eLibrary full-text, contact MGMSU/BelMAPO.")
    asia_count = sum(1 for r in unique if (r.get("region_origin") or "").lower() in {"china", "vietnam", "india", "pakistan", "iran", "saudi arabia", "thailand", "malaysia", "indonesia", "philippines"})
    if asia_count < 5:
        gaps.append("- Asian regional coverage thin — iter 2: native-language queries (zh/vi/hi/fa) on CNKI/OpenDataLab/ModelScope.")
    if not gaps:
        gaps.append("- No major gaps identified in basic dimensions — iter 2 should focus on depth/request-based sets.")
    report.extend(gaps)

    report.append("\n## Recommendations for iteration 2")
    report.append("1. **Request-access batch**: MIH-CNN (Schwendicke), Osaka prosthesis (Takahashi), MeMoSA, SegmentAnyTooth, PKU tooth wear. Draft form letters.")
    report.append("2. **Roboflow deep-dive**: drill per-project pages for class:bracket/aligner/filling/crown/fluorosis/erosion — iter_1 hit 503s; retry with rotation.")
    report.append("3. **Native-language expansion**: zh/vi/es/pt/hi/fa/ar queries on regional repositories (CNKI, OpenDataLab, ScienceDB, SciELO Brazil).")
    report.append("4. **Russian institutional contact**: hub.sfedu.ru, eLibrary.ru full-text, MGMSU Evdokimov, RUDN, BelMAPO, KazNMU.")
    report.append("5. **Journal Data-Availability crawl** 2024-2026: AJODO, Pediatr Dent, J Prosthet Dent, J Dent, Caries Research, Oral Oncol.")
    report.append("6. **SDF / stainless-steel-crown / Hall technique**: no public photo set — try AAPD, medRxiv, WHO/GSK oral-health photo releases.")

    report.append("\n## Agent-level findings summary\n")
    by_agent = defaultdict(list)
    for r in unique:
        for a in r.get("_agent_origins", []):
            by_agent[a].append(r)
    for agent, recs in sorted(by_agent.items()):
        report.append(f"### {agent} ({len(recs)} datasets)")
        for r in recs[:10]:
            ni = r.get("num_images")
            ni_str = f"{ni:,} imgs" if isinstance(ni, int) else "n/a"
            report.append(f"- **{r.get('id')}** — {r.get('title')} ({ni_str}, {r.get('license') or '—'})")
        report.append("")

    with open(f"{ITER_DIR}/REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(json.dumps({
        "raw": len(raw),
        "unique": m["total_unique"],
        "new_vs_master": len(new_unique),
        "stop_recommended": cfg["stop_recommended"],
        "total_images": m["total_images"],
        "by_agent": m["by_agent_origin"],
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
