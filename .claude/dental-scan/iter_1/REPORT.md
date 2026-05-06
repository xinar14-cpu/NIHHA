# Iteration 1 — Dental Dataset Scan REPORT

- Input files: 9
- Raw records: **69**
- Unique (deduped): **54**
- New vs master_manifest: **54**
- Total images across unique datasets: **222,607**
- stop_recommended (next iter): **False**

## Coverage by agent origin
| Agent | Unique records contributed |
|---|---|
| asia-latam-africa | 14 |
| disease-perio | 9 |
| view-protocol | 8 |
| disease-soft-tissue | 8 |
| disease-noncarious | 7 |
| restorative | 6 |
| ru-cis | 5 |
| ortho-pediatric | 4 |
| disease-caries | 3 |

## Coverage by region
| Region | Count |
|---|---|
| unknown | 13 |
| Russia | 7 |
| China | 5 |
| India | 4 |
| Vietnam | 3 |
| Bangladesh | 2 |
| multi | 2 |
| Pakistan (Mithi, Sindh) | 1 |
| South Korea | 1 |
| Germany | 1 |
| Europe | 1 |
| China (Nanjing) | 1 |
| Sri Lanka | 1 |
| Egypt (Cairo University) | 1 |
| India (Ragas Dental College / multi-site) | 1 |
| 5 countries (multi-country) | 1 |
| India (Karnataka) | 1 |
| Thailand | 1 |
| Japan | 1 |
| China (Soochow University, Changzhou) | 1 |
| China (CVPR 2024 Zou et al.) | 1 |
| multi-country (5) | 1 |
| Egypt | 1 |
| Iran | 1 |
| Saudi Arabia | 1 |

## License distribution
| License | Count |
|---|---|
| CC BY 4.0 | 11 |
| CC BY-NC-ND 4.0 | 5 |
| varies | 5 |
| article CC BY | 3 |
| Apache-2.0 | 2 |
| on_request | 2 |
| unknown_paper_only | 2 |
| CC BY-SA 4.0 | 1 |
| CC BY-NC 4.0 | 1 |
| kaggle_terms | 1 |
| MIT | 1 |
| pending | 1 |
| CC BY 4.0 (verify) | 1 |
| CC BY 4.0 (non-commercial access per request) | 1 |
| unspecified (per-workbench access) | 1 |
| Roboflow Public (CC BY 4.0 typical) | 1 |
| varies_per_project | 1 |
| not_specified | 1 |
| not_specified_research_use | 1 |
| roboflow_public_default | 1 |
| research_only_on_request | 1 |
| research_only | 1 |
| credentialed_access | 1 |
| research | 1 |
| Apache-2.0 (code) / open license on dataset | 1 |
| open (Kaggle competition terms) | 1 |
| unknown (not publicly released yet) | 1 |
| open after patent | 1 |
| unspecified article supplementary | 1 |
| open-source framework code plus weights | 1 |
| Indian government open-data terms TBD | 1 |

## Views coverage
| View | Datasets hitting view |
|---|---|
| frontal | 32 |
| upper_occlusal | 20 |
| lower_occlusal | 20 |
| left_lateral | 13 |
| right_lateral | 13 |
| intraoral_clinical | 7 |
| lateral | 2 |
| macro_close_up | 2 |
| close_up_lesion | 2 |
| close_up | 1 |
| unknown | 1 |
| buccal | 1 |
| tongue | 1 |
| palate | 1 |
| smartphone_freehand | 1 |
| full_mouth | 1 |
| occlusal | 1 |
| multiple | 1 |
| multiple_clinical_views | 1 |
| mixed_clinical | 1 |
| frontal_occlusion | 1 |
| right_buccal | 1 |
| left_buccal | 1 |
| clinical_photos | 1 |

## Requires-request vs open-download
- Open download: 33
- Requires request: 21

## Gap analysis
- No SDF-treated pediatric photos found — search iter 2 with explicit AAPD/medRxiv queries.

## Recommendations for iteration 2
1. **Request-access batch**: MIH-CNN (Schwendicke), Osaka prosthesis (Takahashi), MeMoSA, SegmentAnyTooth, PKU tooth wear. Draft form letters.
2. **Roboflow deep-dive**: drill per-project pages for class:bracket/aligner/filling/crown/fluorosis/erosion — iter_1 hit 503s; retry with rotation.
3. **Native-language expansion**: zh/vi/es/pt/hi/fa/ar queries on regional repositories (CNKI, OpenDataLab, ScienceDB, SciELO Brazil).
4. **Russian institutional contact**: hub.sfedu.ru, eLibrary.ru full-text, MGMSU Evdokimov, RUDN, BelMAPO, KazNMU.
5. **Journal Data-Availability crawl** 2024-2026: AJODO, Pediatr Dent, J Prosthet Dent, J Dent, Caries Research, Oral Oncol.
6. **SDF / stainless-steel-crown / Hall technique**: no public photo set — try AAPD, medRxiv, WHO/GSK oral-health photo releases.

## Agent-level findings summary

### asia-latam-africa (14 datasets)
- **zenodo-14827784** — Annotated intraoral image dataset for dental caries detection (6,313 imgs, CC BY-NC-ND 4.0)
- **mendeley-9jnf2jvghy** — Caries-Spectra: A dataset of Enamel Caries (2,000 imgs, CC BY 4.0)
- **mendeley-3253gj88rr** — Gingivitis Image Captioning dataset (Hanoi Medical University) (1,096 imgs, CC BY-NC 4.0)
- **kaggle-salmansajid05-oral-diseases** — Kaggle Oral Diseases dataset (6-class incl. gingivitis/calculus/tartar) (12,653 imgs, kaggle_terms)
- **mendeley-mhjyrn35p4** — Oral Images Dataset (Chandrashekar et al.) - mobile + intraoral camera benign vs malignant (323 imgs, CC BY 4.0)
- **bdj-cairo-9201** — Annotated clinical image dataset for AI classification of malignant and potentially malignant oral lesions (9,201 imgs, unspecified article supplementary)
- **arxiv-2511.04948-code** — COde benchmark multimodal oro-dental dataset for large vision-language models (50,000 imgs, CC BY-NC-ND 4.0)
- **dib-bangladesh-6class** — A comprehensive dental dataset of six classes for deep learning based object detection study (n/a, CC BY 4.0)
- **bmc-shahid-beheshti-3215** — Occlusal intraoral photograph dataset for tooth detection and FDI numbering mixed and permanent dentition (3,215 imgs, article CC BY)
- **mdpi-diagnostics-ksu-435** — Pediatric intraoral photograph dataset for lightweight dental caries screening (435 imgs, article CC BY)

### disease-caries (3 datasets)
- **zenodo-14827784** — Annotated intraoral image dataset for dental caries detection (6,313 imgs, CC BY-NC-ND 4.0)
- **mendeley-9jnf2jvghy** — Caries-Spectra: A dataset of Enamel Caries (2,000 imgs, CC BY 4.0)
- **arxiv-2507.22512-alphadent** — AlphaDent: A dataset for automated tooth pathology detection (1,200 imgs, CC BY-SA 4.0)

### disease-noncarious (7 datasets)
- **alphadent-zftu-v1.1** — AlphaDent: A dataset for automated tooth pathology detection (1,320 imgs, Apache-2.0)
- **pku-tooth-wear-grading-388** — Deep learning-based tooth wear severity grading system — 388 intraoral photos (388 imgs, unknown_paper_only)
- **mendeley-6zsnhrds9t-noncarious-teeth** — Teeth or Dental image dataset (noncarious teeth, children 1-14y) (9,562 imgs, CC BY 4.0)
- **mih-cnn-springer-2022-3241** — Intraoral photograph dataset for AI-based MIH diagnostics (3241 images) (3,241 imgs, unknown_paper_only)
- **github-yunwu2024-dental-fluorosis** — Dental fluorosis image dataset (FusionDentNet) (n/a, pending)
- **figshare-19641750-fluorosis-raw** — Raw data for dental fluorosis (n/a, CC BY 4.0 (verify))
- **zenodo-14622450-athletes-caries-erosion** — Dental caries, tooth erosion and nutritional habits in a cohort of athletes (n/a, CC BY 4.0)

### disease-perio (9 datasets)
- **mendeley-3253gj88rr** — Gingivitis Image Captioning dataset (Hanoi Medical University) (1,096 imgs, CC BY-NC 4.0)
- **oral-mamba-liu-2024** — Oral-Mamba segmentation dataset (Liu et al., BMC Oral Health 2024) (3,365 imgs, CC BY-NC-ND 4.0)
- **alphadent-2025** — AlphaDent instance-segmentation intraoral dataset (1,320 imgs, Apache-2.0)
- **roboflow-gingivitis-fdams** — Roboflow gingivitis-dataset-fdams (segmentation) (162 imgs, varies)
- **roboflow-gingivitis-t98xc** — Roboflow digital-health-bg/gingivitis-t98xc (detection) (204 imgs, varies)
- **kaggle-salmansajid05-oral-diseases** — Kaggle Oral Diseases dataset (6-class incl. gingivitis/calculus/tartar) (12,653 imgs, kaggle_terms)
- **github-pknu-calculus** — PKNU-PR-ML-Lab calculus detection repo (n/a, MIT)
- **fdtooth-scidata-2025** — FDTooth intraoral + CBCT (Sci Data 2025) — perio subset (gingival recession) (1,800 imgs, CC BY 4.0)
- **mdpi-diagnostics-plaque-oleary** — MDPI Diagnostics plaque/O'Leary 3-stage smartphone set (531 imgs, on_request)

### disease-soft-tissue (8 datasets)
- **github-autooral-2024** — Autooral: oral ulcer multi-task (segmentation + classification) clinical image dataset (420 imgs, CC BY-NC-ND 4.0)
- **zenodo-10664056** — A comprehensive dataset of annotated oral cavity images for diagnosis of oral cancer and oral potentially malignant disorders (3,000 imgs, CC BY-NC-ND 4.0)
- **zenodo-14571990** — An Annotated Clinical Image Dataset for Deep Learning-Based Classification of Oral Lesions (Cairo University) (9,201 imgs, CC BY 4.0 (non-commercial access per request))
- **figshare-smartom-31341790** — SMART-OM: Smartphone-based Expert Annotated Dataset of Oral Mucosa images (2,469 imgs, CC BY 4.0)
- **memosa-dataset-2026** — MeMoSA: multi-country collection of over 30,000 oral mucosa images with clinically labelled lesions (30,039 imgs, unspecified (per-workbench access))
- **mendeley-mhjyrn35p4** — Oral Images Dataset (Chandrashekar et al.) - mobile + intraoral camera benign vs malignant (323 imgs, CC BY 4.0)
- **roboflow-oral-lesion2** — Oral lesion2 object detection dataset (benign / precancerous) (39 imgs, Roboflow Public (CC BY 4.0 typical))
- **nature-s41597-024-04099-x-multispectral** — In-vivo non-contact multispectral oral disease image dataset with segmentation (n/a, CC BY 4.0)

### ortho-pediatric (4 datasets)
- **mendeley-6zsnhrds9t-noncarious-teeth** — Teeth or Dental image dataset (noncarious teeth, children 1-14y) (9,562 imgs, CC BY 4.0)
- **github-omni-2025** — OMNI: Oral and Maxillofacial Natural Images Dataset for Malocclusion Assessment (4,166 imgs, not_specified)
- **teethseg-io150k-rgb08k** — IO150K (RGB0.8K subset only) - Intraoral RGB photos for orthodontic tooth instance segmentation (800 imgs, not_specified_research_use)
- **roboflow-orthodontic-noybl** — Orthodontic Object Detection Dataset (Roboflow Universe) (218 imgs, roboflow_public_default)

### restorative (6 datasets)
- **alphadent-2025** — AlphaDent instance-segmentation intraoral dataset (1,320 imgs, Apache-2.0)
- **osaka-prosthesis-1904-takahashi-2021** — Osaka University 1904-image prosthesis classification dataset (Takahashi et al., Sci Rep 2021) (1,904 imgs, on_request)
- **roboflow-havij-caries-tsrca** — Roboflow Universe CaRies Instance Segmentation (Havij) (n/a, varies)
- **roboflow-dental-mate-crown-detection** — Roboflow Dental Mate — Crown Detection (362 imgs, varies)
- **roboflow-ai-dentistry-yang-dental** — AI in Dentistry — Yang Dental intraoral cam instance segmentation (n/a, varies)
- **roboflow-restoration-filling-aggregate** — Roboflow Universe aggregate — class:filling + class:restoration + class:veneer + class:implant_crown index (n/a, varies_per_project)

### ru-cis (5 datasets)
- **alphadent-zftu-v1.1** — AlphaDent: A dataset for automated tooth pathology detection (1,320 imgs, Apache-2.0)
- **alphadent-arxiv-2507-22512** — AlphaDent: A dataset for automated tooth pathology detection (1,320 imgs, Apache-2.0 (code) / open license on dataset)
- **alphadent-kaggle-competition** — AlphaDent: Teeth marking (Kaggle Competition) (1,320 imgs, open (Kaggle competition terms))
- **sfedu-yufu-caries-smartphone-2026** — YuFU/SFedU smartphone caries detection prototype dataset (n/a, unknown (not publicly released yet))
- **sechenov-minzdrav-sjogren-salivary-2024** — Sechenov University first Minzdrav AI-platform dataset: Sjogren syndrome salivary gland pathology (n/a, open after patent)

### view-protocol (8 datasets)
- **fdtooth-scidata-2025** — FDTooth intraoral + CBCT (Sci Data 2025) — perio subset (gingival recession) (1,800 imgs, CC BY 4.0)
- **roboflow-restoration-filling-aggregate** — Roboflow Universe aggregate — class:filling + class:restoration + class:veneer + class:implant_crown index (n/a, varies_per_project)
- **segmentanytooth-hcmc** — SegmentAnyTooth multi-view intraoral dataset (HCMC UMP, Vietnam) (5,000 imgs, research_only_on_request)
- **mendeley-6zsnhrds9t** — Teeth or Dental image dataset (pediatric, 1-14 y/o) (9,562 imgs, CC BY 4.0)
- **mendeley-3253gj88rr** — Gingivitis image-captioning dataset (1,096 imgs, CC BY 4.0)
- **smart-om-smartphone-oral** — SMART-OM smartphone oral imaging dataset (2,469 imgs, research_only)
- **memosa-multi-country-oral** — MeMoSA mobile oral mucosa screening archive (30,039 imgs, credentialed_access)
- **teethdreamer-5photo-protocol** — TeethDreamer 5-view intraoral + 3D mesh pairs (n/a, research)

