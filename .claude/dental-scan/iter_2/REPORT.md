# Iteration 2 — Dental Dataset Scan REPORT

- Input files: 9
- Raw records: **51**
- Unique (deduped): **46**
- New vs master_manifest: **46**
- Total images across unique datasets: **95,147**
- stop_recommended (next iter): **False**

## Coverage by agent origin
| Agent | Unique records contributed |
|---|---|
| disease-soft-tissue | 9 |
| disease-caries | 8 |
| disease-perio | 8 |
| restorative | 6 |
| ortho-pediatric | 5 |
| disease-noncarious | 5 |
| asia-latam-africa | 5 |
| view-protocol | 3 |
| ru-cis | 1 |

## Coverage by region
| Region | Count |
|---|---|
| unknown | 21 |
| China | 4 |
| Saudi Arabia (King Faisal University, Faris Asiri) | 1 |
| unknown (student project) | 1 |
| Pakistan (student team) | 1 |
| unknown (re-upload of Kaggle data) | 1 |
| USA/China (UCLA + Nanjing Medical) | 1 |
| China (Nanjing) | 1 |
| Hong Kong/Guangdong | 1 |
| Mexico (Guanajuato) | 1 |
| USA_Colorado | 1 |
| China_central | 1 |
| India (ICMR Ragas Chennai) | 1 |
| Japan | 1 |
| n_a | 1 |
| Hong Kong (HKU Faculty of Dentistry, 2 centers 2010-2023) | 1 |
| Germany | 1 |
| USA | 1 |
| China (Shanghai Stomatological Hospital) | 1 |
| Peru (EsSALUD Lima) | 1 |
| Russia | 1 |
| unknown_mixed | 1 |
| Brazil | 1 |

## License distribution
| License | Count |
|---|---|
| CC BY 4.0 | 15 |
| request_only | 5 |
| CC_BY_4.0 | 3 |
| kaggle_default | 3 |
| unspecified | 2 |
| request-based | 1 |
| MIT | 1 |
| unknown_roboflow_public | 1 |
| protocol_stage_not_released | 1 |
| CC_BY_article_only | 1 |
| data_on_request | 1 |
| CC_BY_NC_ND | 1 |
| Dryad_default | 1 |
| HF_default | 1 |
| DUA_required | 1 |
| restricted_academic | 1 |
| not_public | 1 |
| n_a | 1 |
| PhysioNet (ODC-BY expected) | 1 |
| AAP_educational_noncommercial_attribution | 1 |
| unknown | 1 |
| platform-gated | 1 |
| unspecified GitHub open | 1 |

## Views coverage
| View | Datasets hitting view |
|---|---|
| frontal | 12 |
| unspecified | 8 |
| left_lateral | 6 |
| right_lateral | 6 |
| upper_occlusal | 6 |
| lower_occlusal | 6 |
| occlusal | 3 |
| close_up_tooth | 2 |
| smartphone_intraoral | 2 |
| upper_anterior_frontal | 1 |
| posterior_occlusal | 1 |
| close_up_intraoral | 1 |
| lateral | 1 |
| mixed | 1 |
| facial_profile_lateral | 1 |
| anterior_deciduous | 1 |
| anterior | 1 |
| selfie_mirror | 1 |
| clinical_intraoral | 1 |
| intraoral_clinical | 1 |
| microscope_10x_40x_HE_stained | 1 |

## Requires-request vs open-download
- Open download: 29
- Requires request: 17

## Gap analysis
- No SDF-treated pediatric photos found — search iter 2 with explicit AAPD/medRxiv queries.
- RU/CIS representation low — iter 2: probe hub.sfedu.ru, eLibrary full-text, contact MGMSU/BelMAPO.
- Asian regional coverage thin — iter 2: native-language queries (zh/vi/hi/fa) on CNKI/OpenDataLab/ModelScope.

## Recommendations for iteration 2
1. **Request-access batch**: MIH-CNN (Schwendicke), Osaka prosthesis (Takahashi), MeMoSA, SegmentAnyTooth, PKU tooth wear. Draft form letters.
2. **Roboflow deep-dive**: drill per-project pages for class:bracket/aligner/filling/crown/fluorosis/erosion — iter_1 hit 503s; retry with rotation.
3. **Native-language expansion**: zh/vi/es/pt/hi/fa/ar queries on regional repositories (CNKI, OpenDataLab, ScienceDB, SciELO Brazil).
4. **Russian institutional contact**: hub.sfedu.ru, eLibrary.ru full-text, MGMSU Evdokimov, RUDN, BelMAPO, KazNMU.
5. **Journal Data-Availability crawl** 2024-2026: AJODO, Pediatr Dent, J Prosthet Dent, J Dent, Caries Research, Oral Oncol.
6. **SDF / stainless-steel-crown / Hall technique**: no public photo set — try AAPD, medRxiv, WHO/GSK oral-health photo releases.

## Agent-level findings summary

### asia-latam-africa (5 datasets)
- **figshare-code-oral-mucosa-30550889** — CODE - Comprehensive Oral Mucosa Database with Explanations (n/a, CC BY 4.0)
- **figshare-siop-29761292** — SIOP dataset: Standardized Periodontal Intraoral Photographic-series (1,350 imgs, CC BY 4.0)
- **datasetninja-dentalai-2495** — DentalAI - intraoral photograph dataset for instance segmentation of caries cavity crack (2,495 imgs, CC BY 4.0)
- **github-tooth-segmentation44-yolov8** — tooth-segmentation44 - YOLOv8 improved dental lesion segmentation Chinese release (n/a, unspecified GitHub open)
- **mendeley-ndb-ufes-bbmmm4wgr8-reference-only** — NDB-UFES Brazil oral cancer leukoplakia histopathology (LATAM reference only; excluded by histology modality) (237 imgs, CC BY 4.0)

### disease-caries (8 datasets)
- **mendeley-phtw6rmwzd-enamel-caries** — Explainable Deep Learning Framework for Automated Classification of Enamel Caries (dataset + code) (2,000 imgs, CC BY 4.0)
- **roboflow-project-group13-dl-caries** — Dental Caries Detection using DL (Project Group13) (1,162 imgs, CC BY 4.0)
- **roboflow-jesse-perring-caries-detection** — caries detection (Jesse Perring) (844 imgs, CC BY 4.0)
- **roboflow-bscs-8th-tooth-cavity-detection** — tooth cavity detection (bscs 8th semester) (312 imgs, CC BY 4.0)
- **roboflow-yolov8-training-tooth-caries-3** — Tooth Caries 3 (YOLO v8 training) (89 imgs, CC BY 4.0)
- **roboflow-jb-okcsk-kaggle-dental-caries** — kaggle dental caries (jb-okcsk) (n/a, CC BY 4.0)
- **kaggle-maazmakhdoom-dental-cavity** — Dental Cavity Detection Dataset (maazmakhdoom) (500 imgs, unspecified)
- **github-dlcariesscreen-ucla** — DLCariesScreen (UCLA, Liangyuan D. et al.) (n/a, request-based)

### disease-noncarious (5 datasets)
- **roboflow-dental-anomaly-6557** — Dental Anomaly Detection Dataset (abrasion/attrition/caries 10-class) (6,557 imgs, unknown_roboflow_public)
- **jmir-e91239-colorado-fluorosis-2026** — Mobile Imaging ML for Caries, Sealants, Fluorosis (Colorado protocol) (n/a, protocol_stage_not_released)
- **frontiers-mih-central-china-1568** — MIH cohort in endemic fluorosis region, central China (1568 schoolchildren) (n/a, CC_BY_article_only)
- **human-tooth-crack-nir-2024** — Human Tooth Crack NIR Imaging Dataset (1,194 imgs, data_on_request)
- **heliyon-tooth-crack-deeplabv3plus-2024** — Semantic segmentation tooth cracks DeepLabv3+ dataset (n/a, CC_BY_NC_ND)

### disease-perio (8 datasets)
- **bmc-s12903-024-04460-x-chronic-gingivitis-683** — Chronic gingivitis identification via transfer ensemble learning (683 intraoral images) (683 imgs, request_only)
- **gdrive-nanjing-oral-endoscopy-3365-mirror** — Oral-Mamba Nanjing GDrive mirror (3365 imgs, calculus/gingivitis/caries) (3,365 imgs, unspecified)
- **chau-hku-gingivitis-567-2023** — Chau HKU/GDUT AI photographic gingivitis detection (567 images) (567 imgs, request_only)
- **terven-ipn-plaque-531-oleary-2025** — Terven IPN Mexico plaque YOLO set (531 RGB, O'Leary, disclosing-gel) (531 imgs, request_only)
- **roboflow-image-segmentation-gingivitis-fdams-v1** — Roboflow image-segmentation-ltmbq Gingivitis Dataset (instance seg) (162 imgs, CC_BY_4.0)
- **roboflow-tooth-ytblb-dental-plaque-v6** — Roboflow tooth-ytblb dental-plaque v6 (70 imgs, CC_BY_4.0)
- **github-priyanshu9898-oral-disease-6class** — Priyanshu9898/Oral-Disease-Classification (6-class, MIT) (n/a, MIT)
- **roboflow-oral-disease-detection-teeth-disease-m1uob-621** — Roboflow teeth-disease-m1uob (621 imgs, calculus/caries/healthy) (621 imgs, CC_BY_4.0)

### disease-soft-tissue (9 datasets)
- **figshare-code-oral-mucosa-30550889** — CODE - Comprehensive Oral Mucosa Database with Explanations (n/a, CC BY 4.0)
- **dryad-modid-multispectral-nvx0k6dxw** — MODID Multispectral Oral Disease (Dryad mirror) (243 imgs, Dryad_default)
- **roboflow-tesisdientes-oral-diseases-4162** — oral-diseases (tesisdientes) (4,162 imgs, CC BY 4.0)
- **huggingface-lines-open-domain-oral-disease-qa** — Open-Domain Oral Disease QA Dataset (n/a, HF_default)
- **kaggle-bavithravairam-mouth-ulcer** — mouth_ulcer (bavithravairam) (n/a, kaggle_default)
- **kaggle-bavithravairam-oral-ulcer** — oral_ulcer (bavithravairam) (n/a, kaggle_default)
- **kaggle-shlokmohanty-ulcer-classification** — Ulcer classification (shlokmohanty) (n/a, kaggle_default)
- **tongue-lesions-bmc-623patients** — Tongue lesions 5-class (BMC Med Imaging 2024) (n/a, request_only)
- **memosa-workbench-access-note** — MeMoSA Workbench access note (operational) (30,039 imgs, DUA_required)

### ortho-pediatric (5 datasets)
- **kaggle-maazmakhdoom-dental-cavity** — Dental Cavity Detection Dataset (maazmakhdoom) (500 imgs, unspecified)
- **fdtooth-physionet-hku-2025** — FDTooth: Intraoral Photographs and CBCT Images for Fenestration and Dehiscence Detection (241 imgs, PhysioNet (ODC-BY expected))
- **mendeley-4mxj6rpv48-facial-profile** — Facial Profile Soft Tissue Annotations for Orthodontic Diagnosis and Classification (400 imgs, CC BY 4.0)
- **schwarzmaier-143-ecc-anterior-deciduous-2024** — 143 intraoral photos of anterior deciduous teeth for ECC external validation (143 imgs, request_only)
- **aap-oral-health-image-gallery-pediatric** — AAP Oral Health Image Gallery (50 imgs, AAP_educational_noncommercial_attribution)

### restorative (6 datasets)
- **roboflow-dental-anomaly-6557** — Dental Anomaly Detection Dataset (abrasion/attrition/caries 10-class) (6,557 imgs, unknown_roboflow_public)
- **oii-ds-benchmark-wang-2023** — OII-DS: A benchmark Oral Implant Image Dataset for object detection and image classification evaluation (19,074 imgs, restricted_academic)
- **osaka-prosthodontics-contact-draft-2026** — Author-contact draft: Osaka Univ Prosthodontics — Takahashi 2021 1904-img restoration dataset (1,904 imgs, not_public)
- **roboflow-intra-oral-scanner-vgggs-r4xwu** — Intra-Oral Scanner classification (intraoral-scanner-vgggs/intra-oral-scanner-r4xwu) (5,758 imgs, CC BY 4.0)
- **roboflow-bitcamp-dental-j1vge** — dental (bitcamp/dental-j1vge) object detection (5,989 imgs, CC BY 4.0)
- **iter2-restorative-gap-note-secondary-caries** — Gap note: no open intraoral dataset for secondary caries / marginal integrity (0 imgs, n_a)

### ru-cis (1 datasets)
- **sechenov-datamed-ai-platform-lead** — Sechenov DataMed.AI / Minzdrav AI platform — institutional lead (no intraoral photo yet) (n/a, platform-gated)

### view-protocol (3 datasets)
- **figshare-siop-29761292** — SIOP dataset: Standardized Periodontal Intraoral Photographic-series (1,350 imgs, CC BY 4.0)
- **zenodo-10580117-teledentistry-peru** — Smartphone teledentistry caries detection dataset (Lima, Peru) (1,305 imgs, CC BY 4.0)
- **jocpd-2026-011-pediatric-5view** — Pediatric intraoral clinical photograph dataset (5-view classification) (3,100 imgs, unknown)

