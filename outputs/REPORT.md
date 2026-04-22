# Dental Dataset Scan — Iteration 4 Report (2026-04-22)

## TL;DR

Across 4 iterations, **150 unique intraoral-photo datasets** were catalogued (154 raw records, deduped by URL + num_images).

**Photos you can download RIGHT NOW (no request, no DUA, no email): ~170,000 across 74 datasets.**
After further manual dedup of 3 known remaining duplicate pairs (Mendeley `6zsnhrds9t` ×2, AlphaDent ×3, Nanjing/OralMamba ×2), the honest figure is **~140–150k photos immediately downloadable**.

Locked behind request/DUA (MeMoSA, AIHub Korea, BDJ Cairo 9201, Bengbu 7200, Charité 5266, OII-DS 15240, CLASEG 2072, BSPC 3246, OralGPT, MIH-3241, etc.): **~180k photos across 67 datasets**.

## Output files (under `outputs/`)

| File | Contents |
|---|---|
| `master_manifest.jsonl` | 150 deduped records (full metadata) |
| `downloadable_now.jsonl` | 74 records you can fetch today |
| `downloadable_now.csv` | same, spreadsheet-friendly |
| `requires_request.csv` | 67 records needing email/DUA/portal access |
| `unclear.csv` | 9 records with ambiguous license/availability |
| `iter_1/…/iter_4/` | raw per-agent JSONL for reproducibility |

## Realistically downloadable now (top 25 datasets, ~158k photos)

| # | num_images | id | source |
|---|---:|---|---|
| 1 | 50,000 | arxiv-2511.04948-code † | arXiv author-hosted (CC BY-NC-ND 4.0) |
| 2 | 12,653 | kaggle-salmansajid05-oral-diseases | Kaggle |
| 3 | 9,562 | mendeley-6zsnhrds9t-noncarious-teeth | Mendeley |
| 4 | 9,562 | mendeley-6zsnhrds9t (dup of #3) | Mendeley |
| 5 | 6,719 | dryad-tcm-tongue-1c59zw48r | Dryad |
| 6 | 6,557 | roboflow-dental-anomaly-6557 | Roboflow |
| 7 | 6,313 | zenodo-14827784 | Zenodo |
| 8 | 6,160 | roboflow-abdul-aziz-caries-tartar-missing-6160 | Roboflow |
| 9 | 5,989 | roboflow-bitcamp-dental-j1vge | Roboflow |
| 10 | 5,758 | roboflow-intra-oral-scanner-vgggs-r4xwu | Roboflow |
| 11 | 5,000 | segmentanytooth-ump-5000 | Mendeley (UMP, Vietnam) |
| 12 | 4,166 | github-omni-2025 | GitHub (OMNI) |
| 13 | 4,162 | roboflow-tesisdientes-oral-diseases-4162 | Roboflow |
| 14 | 3,405 | roboflow-dental-plaque-sorbonne-teeth-detection-xdkru-3405 | Roboflow (Sorbonne) |
| 15 | 3,365 | oral-mamba-liu-2024 / gdrive-nanjing-mirror (same) | GDrive |
| 16 | 2,495 | datasetninja-dentalai-2495 | Supervisely |
| 17 | 2,469 | figshare-smartom-31341790 | Figshare (SMART-OM Korea) |
| 18 | 2,000 | mendeley-9jnf2jvghy | Mendeley |
| 19 | 2,000 | mendeley-phtw6rmwzd-enamel-caries | Mendeley |
| 20 | 1,350 | figshare-siop-29761292 | Figshare |
| 21 | 1,320 | alphadent-2025 (triplicate; one fetch) | arXiv/GitHub/Kaggle |
| 22 | 1,305 | zenodo-10580117-teledentistry-peru | Zenodo (Peru) |

† **arxiv-2511.04948-code 50k** — verify before trusting: arXiv abstract claims 50k usable intraoral RGB images from 8775 checkups, CC BY-NC-ND 4.0; "arXiv author-hosted" is not a stable data host — may be on GitHub/HF linked in the paper.

## By source platform (downloadable-now, 74 datasets)

| Platform | Count |
|---|---:|
| Roboflow Universe | 22 |
| Kaggle | 8 |
| Mendeley Data | 8 |
| GitHub | 4 |
| Figshare | 4 |
| Zenodo | 3 |
| Dryad | 2 |
| Dataset Ninja / Supervisely | 1 |
| Google Drive mirrors | 2 |
| arXiv / HF mixed | ~3 |

## Locked (requires request/DUA) — top strategic targets

| num_images | dataset | access path |
|---:|---|---|
| 30,039 | MeMoSA (Malaysia-led, 5 countries) | workbench.memosa.my + Sci Data 2026 authors |
| 15,240 | OII-DS Subset-B (implants) | Comput Biol Med 2023 author Wang |
| 9,201 | BDJ Cairo oral risk (same as zenodo-14571990) | BDJ 2025 corresponding author |
| 7,671 | Sun Yat-sen pediatric multi-view | PMC12905764 PI |
| 7,200 | Bengbu occlusion | PeerJ 20140 / PMC12478310 |
| 5,854 | MDPI Dentistry malocclusion 17-class IOTN | dj14010060 author |
| 5,266 | Charité Berlin Angle classification | BMC OH 07550-6 author |
| 3,246 | BSPC 40-class mucosa | BSPC 108481 author |
| 3,241 | MIH-3241 CNN (Clin Oral Invest 2022) | 10.1007/s00784-022-04552-4 |
| 3,215 | Shahid Beheshti tooth detection | BMC OH 05803-y author |
| 3,100 | JOCPD pediatric 5-view | 10.22514/jocpd.2026.011 author |
| 3,000 | Zenodo 10664056 annotated oral | institutional email |
| 2,467 | BMC mixed dentition MIH/fluorosis/hypoplasia | BMC OH 06866-7 author |
| 2,072 | CLASEG 16-class | Sci Rep 2025 03268-1 author |
| 1,394 | Kim QH plaque | BMC OH 06350-2 Kim JH |
| 1,194 | Human tooth crack NIR | Ann Biomed Eng 2024 author |
| 1,139 | OralGPT DFull+DPartial (private) | arXiv 2510.13911 Xi'an Jiaotong |
| 600 | Nantakeeratipat plaque Thailand | teerachate@g.swu.ac.th |
| AIHub | SNU Korea intraoral | AIHub portal form |

**67 locked datasets total, ~180k photos.**

## Iteration 4 deltas (what this iter added)

| agent | new/useful records | notable |
|---|---:|---|
| 02 perio | 3 in-scope + 6 saturation/gap notes | Aakash calculus 927, Abdul-Aziz tartar 6160, Nanchang 3869 (saturation: new_unique=3 < threshold=5) |
| 03 noncarious (endemic/developmental refocus) | 10 | Mexican fluorosis-MIH 573, Vietnam MIH 1834 / Dong Thap, Saudi Abha 520, Portuguese tooth wear, MDPI JCM 8959 hypoplasia |
| 04 soft-tissue (retry after timeout) | 9 | CLASEG-2072 resolved, BSPC-3246 resolved, OralGPT confirmed private, Autooral 420 **public** (Sci Rep 2024) |
| 10 surgical (new zone, first scan) | 5 | ZONE GAP confirmed — phlegmon / MRONJ / fistula / pericoronitis structurally absent from open ML datasets |

## Recommended immediate next steps

1. **Actually download the top-25 "downloadable now" list above** — script it via `wget`/Roboflow-API/Kaggle-API. True cumulative after manual dedup ≈ 140–150k photos. This covers caries, perio, plaque, malocclusion, oral lesions, implant/restoration views.
2. **Verify `arxiv-2511.04948-code 50000`** — inspect the paper for actual hosting URL (GitHub/HF), not just arXiv.
3. **Harvest PMC CC-BY case-report figures** for the confirmed gaps — angular cheilitis, denture stomatitis, hairy tongue, MRONJ, phlegmon, fistula, pericoronitis, dry socket — these only exist in individual figures, no dedicated open datasets.
4. **Outreach to the ~19 top locked datasets** (separate email workflow if needed).

## Saturation signals

- Caries / perio / basic malocclusion: **saturated** (new_unique < 5 per iter).
- Noncarious endemic fluorosis outside Western sources: **thin but growing** (Mexico, Saudi, Vietnam filled iter-4).
- Surgical / acute / MRONJ / phlegmon: **structural gap** in open ML datasets — not just a search gap.
- Benign inflammatory mucosa (angular cheilitis etc.): **structural gap**; exists only inside CLASEG / BSPC multi-class sets.
