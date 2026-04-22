# Iteration 3 — Dental Dataset Scan REPORT

- Input files: 9
- Raw records: **60**
- Unique (deduped): **53**
- New vs master_manifest: **53**
- Total images across unique datasets: **516,768**
- stop_recommended (next iter): **False**

## Coverage by agent origin
| Agent | Unique records contributed |
|---|---|
| disease-perio | 12 |
| disease-noncarious | 10 |
| disease-soft-tissue | 9 |
| restorative | 7 |
| view-protocol | 5 |
| ortho-pediatric | 4 |
| disease-caries | 3 |
| asia-latam-africa | 3 |
| ru-cis | 2 |

## Coverage by region
| Region | Count |
|---|---|
| unknown | 11 |
| China | 7 |
| South Korea | 2 |
| Russia | 2 |
| South Korea (Seoul National University Dental Hospital) | 1 |
| 21 countries | 1 |
| France (Sorbonne) | 1 |
| India (likely) | 1 |
| Vietnam (AIRC Hanoi) | 1 |
| Philippines (likely) | 1 |
| USA Michigan | 1 |
| Taiwan NYCU | 1 |
| Thailand | 1 |
| China_Guizhou_endemic_coal_fluorosis | 1 |
| Germany_LMU_Munich | 1 |
| Europe_unspecified | 1 |
| multi_country_review | 1 |
| Europe_clinical | 1 |
| France_O-Rares_reference_centre | 1 |
| Slovenia | 1 |
| referral_hospital | 1 |
| Bangladesh + global web | 1 |
| Europe multi-country consortium | 1 |
| China Xi'an Jiaotong Stomatology Hospital | 1 |
| Japan | 1 |
| Pakistan/Thailand/USA | 1 |
| USA (NYC public schools) | 1 |
| China (Bengbu) | 1 |
| global | 1 |
| Germany (Charité Berlin) | 1 |
| China (Sun Yat-sen Hospital of Stomatology + Guangzhou CDC) | 1 |
| China (First Affiliated Hospital Bengbu Medical U.) | 1 |
| international (Uribe et al., Sage JDR 2024) | 1 |
| Indonesia | 1 |
| Georgia | 1 |

## License distribution
| License | Count |
|---|---|
| unknown | 11 |
| upon_request | 4 |
| CC_BY_4.0 | 4 |
| restricted | 3 |
| CC BY 4.0 | 2 |
| on_request | 2 |
| CC_BY | 2 |
| CC BY 4.0 (Roboflow default, verify) | 2 |
| not_specified | 2 |
| AIHub Korea DUA (approval required) | 1 |
| CC BY 4.0 (to verify) | 1 |
| CC BY 4.0 article | 1 |
| CC_BY_NC_article_only | 1 |
| upon_request_likely | 1 |
| AGPL-3.0 | 1 |
| repo_code_license_unspecified_dataset | 1 |
| closed_on_request | 1 |
| CC BY 4.0 (per arXiv; verify Dryad CC0 default) | 1 |
| CC BY 4.0 (SciDB default, verify) | 1 |
| CC BY-NC-SA 4.0 | 1 |
| Unclear/copyrighted; bulk-export UNVERIFIED | 1 |
| Not released; author contact | 1 |
| Unverified; DAS inaccessible | 1 |
| Private clinical; not released | 1 |
| Apache-2.0 | 1 |
| AI-Hub custom (non-commercial default) | 1 |
| Restricted (Data Request Form required) | 1 |
| CC-BY 4.0 paper; dataset NOT released | 1 |
| Russian state DB reg 2023624045; access by request | 1 |
| Russian state DB reg; request-only | 1 |

## Views coverage
| View | Datasets hitting view |
|---|---|
| frontal | 23 |
| left_lateral | 10 |
| right_lateral | 10 |
| occlusal | 9 |
| upper_occlusal | 5 |
| lower_occlusal | 5 |
| lateral | 5 |
| tongue_protruded | 2 |
| buccal | 2 |
| tongue | 2 |
| lip | 2 |
| palate | 2 |
| full_mouth_mixed | 2 |
| smile | 1 |
| review_aggregate | 1 |
| buccal_sextant | 1 |
| anterior | 1 |
| frontal_anterior_teeth | 1 |
| occlusal_close_up | 1 |
| close_up | 1 |
| panoramic_paired | 1 |
| close-up_lesion | 1 |
| skin_lesion | 1 |
| perioral_possible | 1 |
| floor | 1 |
| lesion_close-up | 1 |
| intraoral_camera | 1 |
| catalogue | 1 |
| upper_anterior | 1 |
| lower_anterior | 1 |
| palatal | 1 |

## Requires-request vs open-download
- Open download: 21
- Requires request: 32

## Gap analysis
- RU/CIS representation low — iter 2: probe hub.sfedu.ru, eLibrary full-text, contact MGMSU/BelMAPO.

## Recommendations for iteration 2
1. **Request-access batch**: MIH-CNN (Schwendicke), Osaka prosthesis (Takahashi), MeMoSA, SegmentAnyTooth, PKU tooth wear. Draft form letters.
2. **Roboflow deep-dive**: drill per-project pages for class:bracket/aligner/filling/crown/fluorosis/erosion — iter_1 hit 503s; retry with rotation.
3. **Native-language expansion**: zh/vi/es/pt/hi/fa/ar queries on regional repositories (CNKI, OpenDataLab, ScienceDB, SciELO Brazil).
4. **Russian institutional contact**: hub.sfedu.ru, eLibrary.ru full-text, MGMSU Evdokimov, RUDN, BelMAPO, KazNMU.
5. **Journal Data-Availability crawl** 2024-2026: AJODO, Pediatr Dent, J Prosthet Dent, J Dent, Caries Research, Oral Oncol.
6. **SDF / stainless-steel-crown / Hall technique**: no public photo set — try AAPD, medRxiv, WHO/GSK oral-health photo releases.

## Agent-level findings summary

### asia-latam-africa (3 datasets)
- **bmc-mixed-dentition-intraoral-2025-06866** — Mixed dentition 5-view multi-class (MIH+fluorosis+hypoplasia+caries) (2,467 imgs, CC_BY_4.0)
- **bmc-hibogi-cimahi-indonesia-3221-2025** — HI Bogi: YOLO-V8x dental caries dataset of Cimahi primary school children (Indonesia) (3,221 imgs, not_specified)
- **bmc-tbilisi-humanitarian-2864-georgia-2025** — Intraoral mobile photography for OH screening in children (Tbilisi, Georgia) (2,864 imgs, not_specified)

### disease-caries (3 datasets)
- **aihub-korea-71509-dental-intraoral-clinical** — AIHub Korea 치과 구내 임상사진 이미지 데이터 (Dental Intraoral Clinical Photo Image Dataset) (n/a, AIHub Korea DUA (approval required))
- **iter3-gap-note-dentex-hhs9g-kaggle-caries-modality-unclear** — Gap note: dentex-hhs9g/kaggle-caries v2 (12,245 images) modality unresolved (12,245 imgs, CC BY 4.0 (to verify))
- **iter3-gap-note-systematic-review-s41746-01818-5-datasets** — Gap note: npj Digital Medicine s41746-025-01818-5 caries-photo datasets (IDs 50 & 65) (437,538 imgs, CC BY 4.0 article)

### disease-noncarious (10 datasets)
- **dfid-mltrmr-guizhou-131** — DFID: First open-source Dental Fluorosis Image Dataset (Guizhou, MLTrMR) (131 imgs, repo_code_license_unspecified_dataset)
- **integrated-fluorosis-grading-bspc-2024** — Integrated fluorosis grading (seg+classify) (n/a, on_request)
- **ai-mih-intraoral-3241-koi-2022** — AI-based MIH diagnosis intraoral (3241) (3,241 imgs, closed_on_request)
- **mdpi-mih-oct-photonics-799-2025** — OCT+photo MIH case-control dataset (n/a, CC_BY)
- **frontiers-dde-mih-fluorosis-scoping-2025-1616109** — DDE scoping review photo atlas (MIH+fluorosis+hypoplasia) (n/a, CC_BY_4.0)
- **coi-skin-dde-pediatric-2025-06326** — DDE+dental anomalies in pediatric skin-disease cohort 71+41 (n/a, CC_BY)
- **frontiers-ai-witkop-ngs-1130175** — Witkop AI clinical photo series O-Rares (NGS+clinical) (n/a, CC_BY_4.0)
- **mdpi-genes-16-822-2025** — AI imaging+genomics Slovenia WES-24 (n/a, CC_BY_4.0)
- **bmc-mixed-dentition-intraoral-2025-06866** — Mixed dentition 5-view multi-class (MIH+fluorosis+hypoplasia+caries) (2,467 imgs, CC_BY_4.0)
- **cleft-mih-6432-coi-2025-06311** — MIH/HSPM in orofacial-cleft cohort (6432 teeth, 290) (n/a, on_request)

### disease-perio (12 datasets)
- **roboflow-dental-plaque-sorbonne-teeth-detection-xdkru-3405** — Dental Plaque Sorbonne — Teeth Detection (C1-C3 calculus, G1 gingivitis, PDI1-3 plaque) (3,405 imgs, CC BY 4.0)
- **kaggle-santhoshsivang-calculus-dataset** — calculus_dataset (santhoshsivang) (n/a, unknown)
- **ssrn-6281532-airc-labden-plaque-ortho** — AIRC-LABDEN: Multi-Modal Plaque in Fixed Ortho Appliance Patients (n/a, unknown)
- **moharrami-ragadio-2024-gingivitis-666-kaggle-lead** — Ragadio 2024 gingivitis 666 Kaggle (slug unresolved) (666 imgs, unknown)
- **outreach-di-gianfilippo-2025-recession-34-contact** — Di Gianfilippo 2025 — 34 recession photos 2018 class (outreach) (34 imgs, upon_request)
- **outreach-lee-nycu-2025-keratinized-gingiva-576** — Lee NYCU 2025 — 576 buccal photos keratinized gingiva (outreach) (576 imgs, CC_BY_NC_article_only)
- **outreach-kim-2025-plaque-quigley-hein-1394** — Kim 2025 BMC OH — 1394 photos QH plaque index (outreach) (1,394 imgs, upon_request)
- **outreach-nantakeeratipat-2024-plaque-600-thailand** — Nantakeeratipat 2024 — 600 plaque photos SWU Thailand (outreach) (600 imgs, upon_request)
- **outreach-vaughan-2025-multi-view-35-perio** — Vaughan 2025 — 35 multi-view periodontal photos (outreach) (35 imgs, upon_request)
- **outreach-wen-2024-gingivitis-826-sci-rep** — Wen 2024 Sci Rep — 826 photos / 8214 teeth MGI grading (outreach) (826 imgs, upon_request_likely)

### disease-soft-tissue (9 datasets)
- **dryad-tcm-tongue-1c59zw48r** — TCM-Tongue: Standardized Tongue Image Dataset with Pathological Annotations (6,719 imgs, CC BY 4.0 (per arXiv; verify Dryad CC0 default))
- **scidb-tongue-inquiry-8417299d** — Tongue Image Dataset with Inquiry Data (1,194 imgs, CC BY 4.0 (SciDB default, verify))
- **roboflow-ulcers-sw4n4** — Ulcers (oral) - Roboflow Universe (n/a, CC BY 4.0 (Roboflow default, verify))
- **mpox-skin-lesion-v2-hfmd-subset** — Mpox Skin Lesion Dataset v2.0 - HFMD subset (161 imgs, CC BY-NC-SA 4.0)
- **opmdcare-atlas-photos-2026** — OPMDCARE Atlas of Photos (n/a, Unclear/copyrighted; bulk-export UNVERIFIED)
- **claseg-oral-lesions-2025-contact-note** — CLASEG 14-class Oral Lesions Dataset - contact required (2,072 imgs, Not released; author contact)
- **multiclass-oral-mucosa-3246-1013-2025-contact-note** — Multiclass Oral Mucosal Lesions Dataset (3246 images, 1013 patients, 40 cat.) (3,246 imgs, Unverified; DAS inaccessible)
- **oralgpt-mucosa-xian-jiaotong-contact-note** — OralGPT DFull+DPartial Oral Mucosa Dataset (Xi'an Jiaotong) (1,139 imgs, Private clinical; not released)
- **roboflow-oralleukoplakia-418** — OralLeukoplakia - Roboflow Universe (418 imgs, CC BY 4.0 (Roboflow default, verify))

### ortho-pediatric (4 datasets)
- **nyu-caredaway-sdf-school-ny-10620** — CariedAway Pragmatic Non-inferiority Trial of SDF (NYU Data Catalog #10620) (n/a, Restricted (Data Request Form required))
- **peerj-20140-bengbu-occlusion-7200** — Bengbu Medical University lateral-view occlusion dataset (7200 intraoral photos) (7,200 imgs, CC-BY 4.0 paper; dataset NOT released)
- **osf-tf5r8-data-share-ortho** — OSF 'data share ortho project' (tf5r8) (n/a, —)
- **iter3-gap-note-ortho-pediatric** — Iter-3 gap note: no NEW public ortho/pediatric intraoral-photo dataset (0 imgs, —)

### restorative (7 datasets)
- **alphadent-v1.2-release-check-negative-2026** — AlphaDent v1.2 release probe — negative (n/a, Apache-2.0)
- **aihub-kr-restorative-probe-2026** — AI Hub Korea restorative-intraoral probe (n/a, AI-Hub custom (non-commercial default))
- **hf-mendeley-roboflow-api-probe-2026** — Direct API probe: HF/Mendeley/Roboflow (negative) (n/a, —)
- **osaka-prosthesis-outreach-completed-2026-04** — Osaka 1904 prosthesis outreach dispatched (1,904 imgs, —)
- **ding-zhejiang-secondary-caries-outreach-2026** — Ding (Zhejiang) secondary-caries outreach (n/a, —)
- **khurshid-bmc-2025-panoramic-prosthesis-out-of-scope-note** — Khurshid 2025 multi-regional prosthesis — OUT-OF-SCOPE (2,235 imgs, —)
- **iter3-restorative-saturation-report** — Iter-3 restorative niche saturation report (n/a, —)

### ru-cis (2 datasets)
- **sechenov-ai-205-adult-dental-photos-polygon** — Sechenov AI #205: Adult intraoral frontal+lateral photos with polygon tooth segmentation (n/a, Russian state DB reg 2023624045; access by request)
- **sechenov-ai-214-pediatric-dental-morbidity-moscow** — Sechenov AI #214: Pediatric dental morbidity dataset (Moscow/MO) (n/a, Russian state DB reg; request-only)

### view-protocol (5 datasets)
- **iter3-gap-note-systematic-review-s41746-01818-5-datasets** — Gap note: npj Digital Medicine s41746-025-01818-5 caries-photo datasets (IDs 50 & 65) (437,538 imgs, CC BY 4.0 article)
- **charite-berlin-5266-angle-lateral-2025** — Charité Berlin lateral intraoral photo corpus for Angle classification (5,266 imgs, restricted)
- **sunyatsen-zengcheng-pediatric-7671-2026** — Sun Yat-sen pediatric multi-view intraoral caries screening dataset (7,671 imgs, restricted)
- **bengbu-7200-occlusion-classification-2025** — Bengbu Medical Univ. 7,200-image intraoral occlusion classification corpus (7,200 imgs, restricted)
- **osf-tgm5n-dental-ai-resources-list** — OSF Datasets for AI Dental Imaging — Resources list (n/a, CC BY 4.0)

