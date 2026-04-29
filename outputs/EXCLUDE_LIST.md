# Dental Dataset Scan — EXCLUDE LIST

Generated 2026-04-29 from 175 deduped records (raw: 185). Use this file as exclusion context for any deep-search agent: any future hit matching one of these IDs, URLs, DOIs, or aliases is a duplicate and should be skipped.

Sections: OPEN (downloadable now), REQUEST (gated by request/DUA), UNCLEAR, GAP_NOTE (zones/datasets explored and confirmed unavailable).

Format per row:
- `ID` — canonical record id (also unique exclusion key)
- `n` — num_images (0 = unknown / paywall / gated)
- `URL` — canonical link
- `aliases` — pipe-separated search keys (DOI, Mendeley/Zenodo/Figshare/Roboflow/Kaggle/PMC/arXiv/GitHub IDs)
- `topics` — coverage tags (caries|perio|noncarious|soft_tissue|ortho|pediatric|restorative|ru_cis|non_western|view_protocol|surgical|misc)
- `notes` — short context, especially for GAP_NOTE entries

---

## OPEN — Downloadable Now (no request, no DUA) (74 entries)

### OPEN / topic = caries (27)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `kaggle-salmansajid05-oral-diseases` | 12653 | https://www.kaggle.com/datasets/salmansajid05/oral-diseases | `kaggle:salmansajid05/oral-diseases` | Needs modality triage — mixed with X-ray possible. Use photo subset only. |
| `roboflow-dental-anomaly-6557` | 6557 | https://universe.roboflow.com/search?q=class:abrasion+-+filling | `-` | Roboflow dental anomaly project 6557 images includes non-carious classes; slug needs manual verification (WebFetch 403). |
| `zenodo-14827784` | 6313 | https://zenodo.org/records/14827784 | `10.5281/zenodo.14827784|zenodo:14827784` | 5 standard views, W/R and W/O-R subfolders. Primary flagship caries intraoral dataset. Mixed + permanent dentitions. Pil |
| `roboflow-abdul-aziz-caries-tartar-missing-6160` | 6160 | https://universe.roboflow.com/search?q=class%3Atartar | `-` | Slug not fully resolved (503). Tartar subset relevant for perio. |
| `roboflow-bitcamp-dental-j1vge` | 5989 | https://universe.roboflow.com/bitcamp/dental-j1vge | `roboflow:bitcamp/dental-j1vge` | modality_mixed:true pending direct verification |
| `gdrive-nanjing-oral-endoscopy-3365-mirror` | 3365 | https://drive.google.com/drive/folders/1vo_qv3EF9eG4Q2dPvtb_rXb4ttBkYJq_ | `10.1186/s12903-024-05072-1` | Same physical set as oral-mamba-liu-2024 (excluded); distinct access URL — flag dedupe |
| `oral-mamba-liu-2024` | 3365 | https://doi.org/10.1186/s12903-024-05072-1 | `10.1186/s12903-024-05072-1` | Largest perio-relevant segmentation corpus found. Google Drive link in paper. |
| `datasetninja-dentalai-2495` | 2495 | https://datasetninja.com/dentalai | `-` | 28904 labeled objects; train/val/test 1991/254/250; 1018 MB Supervisely format; CC BY 4.0; author-region attribution unc |
| `mendeley-9jnf2jvghy` | 2000 | https://data.mendeley.com/datasets/9jnf2jvghy/2 | `10.17632/9jnf2jvghy.2|mendeley:9jnf2jvghy` | 2000 low-res 224x224 JPG intraoral images: 800 advanced + 800 early-stage + 400 no caries. Classification only (no bbox/ |
| `mendeley-phtw6rmwzd-enamel-caries` | 2000 | https://data.mendeley.com/datasets/phtw6rmwzd/1 | `10.17632/phtw6rmwzd.1|mendeley:phtw6rmwzd` | 2000 intraoral photos 224x224 JPG, 3 class folders, includes EnamelCaries_Experiment-II.py code. Low-res but bona-fide i |
| `alphadent-zftu-v1.1` | 1320 | https://github.com/ZFTurbo/AlphaDent | `github:ZFTurbo/AlphaDent` | DSLR intraoral macro (Canon 6D mk II + 100mm macro + intraoral mirror). Includes wear facets labels — directly in non-ca |
| `zenodo-10580117-teledentistry-peru` | 1305 | https://zenodo.org/doi/10.5281/zenodo.10580117 | `10.5281/zenodo.10580117|zenodo:10580117` | 87 patients × 15 intraoral photos (3 per view × 5 views) captured by family members using Xiaomi Redmi 9A + handheld mir |
| `arxiv-2507.22512-alphadent` | 1200 | https://arxiv.org/abs/2507.22512 | `10.48550/arXiv.2507.22512|arxiv:2507.22512` | 1200+ DSLR photos (Canon 6D mark II + 100mm macro + intraoral mirror) from 295 patients, Jan 2024-Jan 2025. Instance seg |
| `roboflow-project-group13-dl-caries` | 1162 | https://universe.roboflow.com/project-group13/dental-caries-detection-using-dl | `roboflow:project-group13/dental-caries-detection-using-dl` | 1162 tooth-decay images, binary labels. 403 blocked direct page fetch; modality photos per project description; verify o |
| `iter3-perio-rvg-v1301-blacklist-confirm` | 1100 | https://github.com/YOLOv8-YOLOv11-Segmentation-Studio/rvg-v1301 | `github:YOLOv8-YOLOv11-Segmentation-Studio/rvg-v1301` | RVG = radio-visio-graphy X-ray, NOT photo. Log to prevent re-chasing. |
| `roboflow-jesse-perring-caries-detection` | 844 | https://universe.roboflow.com/jesse-perring/caries-detection-pqmf0-io8pp | `roboflow:jesse-perring/caries-detection-pqmf0-io8pp` | 844 caries images per Roboflow listing; 403 blocked direct fetch; flagged intraoral pending visual verification. |
| `roboflow-oral-disease-detection-teeth-disease-m1uob-621` | 621 | https://universe.roboflow.com/oral-disease-detection/teeth-disease-m1uob | `roboflow:oral-disease-detection/teeth-disease-m1uob` | Pre-trained model bundled |
| `kaggle-maazmakhdoom-dental-cavity` | 500 | https://www.kaggle.com/datasets/maazmakhdoom/dental-cavity-detection-dataset | `kaggle:maazmakhdoom/dental-cavity-detection-dataset` | Referenced by MDPI Diagnostics 2026 (16/6/862) pediatric aggregation where 80 pediatric photos were extracted. Filter to |
| `roboflow-bscs-8th-tooth-cavity-detection` | 312 | https://universe.roboflow.com/bscs-8th-semester/tooth-cavity-detection | `roboflow:bscs-8th-semester/tooth-cavity-detection` | 312 images with cavity bounding boxes; published ~2 years ago. Verified via WebSearch only; Roboflow page still returns  |
| `roboflow-yolov8-training-tooth-caries-3` | 89 | https://universe.roboflow.com/yolo-v8-training/tooth-caries-3 | `roboflow:yolo-v8-training/tooth-caries-3` | Small (89 images, meets min>=50). Published June 2024. Useful as augmentation. |
| `aap-oral-health-image-gallery-pediatric` | 50 | https://www.aap.org/en/patient-care/oral-health/oral-health-image-gallery/ | `-` | ~50 imgs estimate. Non-commercial-only. |
| `dib-bangladesh-6class` | 0 | https://www.sciencedirect.com/science/article/pii/S2352340924009326 | `10.1016/j.dib.2024.110926` | Multi-source Bangladesh Prescription Point LabAid IbnSina; 232 patients; N_images to confirm; usable_subset intraoral ph |
| `github-priyanshu9898-oral-disease-6class` | 0 | https://github.com/Priyanshu9898/Oral-Disease-Classification | `github:Priyanshu9898/Oral-Disease-Classification` | GDrive file 13NFWEqtL_3Vxsr02ehXoYTapp63dMIDM; perio classes = calculus+gingivitis |
| `roboflow-dental-qlzfr-dental-problems` | 0 | https://universe.roboflow.com/dental-qlzfr/dental-problems | `roboflow:dental-qlzfr/dental-problems` | Class pattern strongly suggests panoramic/periapical X-ray — needs manual verification; probable BLACKLIST if radiograph |
| `roboflow-havij-caries-tsrca` | 0 | https://universe.roboflow.com/havij/caries-tsrca-guzap | `roboflow:havij/caries-tsrca-guzap` | Modality mixed — intraoral subset must be filtered. Good for pre-training after triage. |
| `roboflow-jb-okcsk-kaggle-dental-caries` | 0 | https://universe.roboflow.com/jb-okcsk/kaggle-dental-caries | `roboflow:jb-okcsk/kaggle-dental-caries` | Jan 2024 re-labelled Kaggle caries dataset. modality_mixed=true since Kaggle source may include X-ray samples. Image cou |
| `zenodo-14622450-athletes-caries-erosion` | 0 | https://zenodo.org/records/14622450 | `10.5281/zenodo.14622450|zenodo:14622450` | Tabular only; no photos. usable_subset=none for image pipeline. |

### OPEN / topic = perio (21)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `kaggle-salmansajid05-oral-diseases` | 12653 | https://www.kaggle.com/datasets/salmansajid05/oral-diseases | `kaggle:salmansajid05/oral-diseases` | Needs modality triage — mixed with X-ray possible. Use photo subset only. |
| `roboflow-abdul-aziz-caries-tartar-missing-6160` | 6160 | https://universe.roboflow.com/search?q=class%3Atartar | `-` | Slug not fully resolved (503). Tartar subset relevant for perio. |
| `roboflow-dental-plaque-sorbonne-teeth-detection-xdkru-3405` | 3405 | https://universe.roboflow.com/dental-plaque-sorbonne/teeth-detection-xdkru | `roboflow:dental-plaque-sorbonne/teeth-detection-xdkru` | 10 classes incl 3-level calculus + 3-level plaque; verify modality on download. |
| `gdrive-nanjing-oral-endoscopy-3365-mirror` | 3365 | https://drive.google.com/drive/folders/1vo_qv3EF9eG4Q2dPvtb_rXb4ttBkYJq_ | `10.1186/s12903-024-05072-1` | Same physical set as oral-mamba-liu-2024 (excluded); distinct access URL — flag dedupe |
| `oral-mamba-liu-2024` | 3365 | https://doi.org/10.1186/s12903-024-05072-1 | `10.1186/s12903-024-05072-1` | Largest perio-relevant segmentation corpus found. Google Drive link in paper. |
| `figshare-siop-29761292` | 1350 | https://figshare.com/articles/dataset/SIOP_dataset/29761292 | `10.6084/m9.figshare.29761292|figshare:29761292` | 150 SPIP series × 9 dentists = 1350 photos; strict 5-view protocol; quality-graded on 5 dimensions |
| `alphadent-2025` | 1320 | https://arxiv.org/abs/2507.22512 | `10.5281/zenodo.16582489|arxiv:2507.22512` | 295 patients DSLR. Commercial-friendly license — rare for perio. Verify which classes are calculus/gingivitis. |
| `alphadent-zftu-v1.1` | 1320 | https://github.com/ZFTurbo/AlphaDent | `github:ZFTurbo/AlphaDent` | DSLR intraoral macro (Canon 6D mk II + 100mm macro + intraoral mirror). Includes wear facets labels — directly in non-ca |
| `arxiv-2507.22512-alphadent` | 1200 | https://arxiv.org/abs/2507.22512 | `10.48550/arXiv.2507.22512|arxiv:2507.22512` | 1200+ DSLR photos (Canon 6D mark II + 100mm macro + intraoral mirror) from 295 patients, Jan 2024-Jan 2025. Instance seg |
| `iter3-perio-rvg-v1301-blacklist-confirm` | 1100 | https://github.com/YOLOv8-YOLOv11-Segmentation-Studio/rvg-v1301 | `github:YOLOv8-YOLOv11-Segmentation-Studio/rvg-v1301` | RVG = radio-visio-graphy X-ray, NOT photo. Log to prevent re-chasing. |
| `mendeley-3253gj88rr` | 1096 | https://data.mendeley.com/datasets/3253gj88rr | `mendeley:3253gj88rr` | Macro frontal anterior — useful for frontal/macro close-up protocol. |
| `roboflow-aakash-npfax-dental-remote-areas-calculus-927` | 927 | https://universe.roboflow.com/aakash-npfax/dental-care-in-remote-areas | `roboflow:aakash-npfax/dental-care-in-remote-areas` | Single-class calculus detection; modality inferred intraoral photo; verify via page visit (503 at scan time). |
| `roboflow-oral-disease-detection-teeth-disease-m1uob-621` | 621 | https://universe.roboflow.com/oral-disease-detection/teeth-disease-m1uob | `roboflow:oral-disease-detection/teeth-disease-m1uob` | Pre-trained model bundled |
| `roboflow-gingivitis-t98xc` | 204 | https://universe.roboflow.com/digital-health-bg/gingivitis-t98xc | `roboflow:digital-health-bg/gingivitis-t98xc` | Small community det set. |
| `roboflow-gingivitis-fdams` | 162 | https://universe.roboflow.com/image-segmentation-ltmbq/gingivitis-dataset-fdams | `roboflow:image-segmentation-ltmbq/gingivitis-dataset-fdams` | Small community seg set. |
| `roboflow-tooth-ytblb-dental-plaque-v6` | 70 | https://universe.roboflow.com/tooth-ytblb/dental-plaque/dataset/6 | `roboflow:tooth-ytblb/dental-plaque` | Auxiliary plaque masks |
| `aap-oral-health-image-gallery-pediatric` | 50 | https://www.aap.org/en/patient-care/oral-health/oral-health-image-gallery/ | `-` | ~50 imgs estimate. Non-commercial-only. |
| `dib-bangladesh-6class` | 0 | https://www.sciencedirect.com/science/article/pii/S2352340924009326 | `10.1016/j.dib.2024.110926` | Multi-source Bangladesh Prescription Point LabAid IbnSina; 232 patients; N_images to confirm; usable_subset intraoral ph |
| `github-pknu-calculus` | 0 | https://github.com/PKNU-PR-ML-Lab/calculus | `github:PKNU-PR-ML-Lab/calculus` | Pusan Nat'l U lab repo. |
| `github-priyanshu9898-oral-disease-6class` | 0 | https://github.com/Priyanshu9898/Oral-Disease-Classification | `github:Priyanshu9898/Oral-Disease-Classification` | GDrive file 13NFWEqtL_3Vxsr02ehXoYTapp63dMIDM; perio classes = calculus+gingivitis |
| `kaggle-santhoshsivang-calculus-dataset` | 0 | https://www.kaggle.com/datasets/santhoshsivang/calculus-dataset | `kaggle:santhoshsivang/calculus-dataset` | Uploaded Jan 2025; modality/license need verification post-download. |

### OPEN / topic = noncarious (10)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `mendeley-6zsnhrds9t-noncarious-teeth` | 9562 | https://data.mendeley.com/datasets/6zsnhrds9t/1 | `10.17632/6zsnhrds9t.1|mendeley:6zsnhrds9t` | Pediatric intraoral, explicitly non-carious. Re-label candidate for MIH/hypoplasia/fluorosis. |
| `roboflow-dental-anomaly-6557` | 6557 | https://universe.roboflow.com/search?q=class:abrasion+-+filling | `-` | Roboflow dental anomaly project 6557 images includes non-carious classes; slug needs manual verification (WebFetch 403). |
| `alphadent-zftu-v1.1` | 1320 | https://github.com/ZFTurbo/AlphaDent | `github:ZFTurbo/AlphaDent` | DSLR intraoral macro (Canon 6D mk II + 100mm macro + intraoral mirror). Includes wear facets labels — directly in non-ca |
| `dfid-mltrmr-guizhou-131` | 131 | https://github.com/uxhao-o/MLTrMR | `github:uxhao-o/MLTrMR` | 26/49/36/20 balance; 560x448->512x512 optical camera; 7:3 split; endemic coal-burning belt. |
| `aap-oral-health-image-gallery-pediatric` | 50 | https://www.aap.org/en/patient-care/oral-health/oral-health-image-gallery/ | `-` | ~50 imgs estimate. Non-commercial-only. |
| `figshare-19641750-fluorosis-raw` | 0 | https://figshare.com/articles/dataset/Raw_data_for_dental_fluorosis/19641750 | `10.6084/m9.figshare.19641750|figshare:19641750` | Fetch returned 503. Content type unconfirmed. Re-verify iter 2. |
| `frontiers-ai-witkop-ngs-1130175` | 0 | https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1130175/full | `10.3389/fphys.2023.1130175` | French reference centre open-access photos of all 4 Witkop types. |
| `frontiers-dde-mih-fluorosis-scoping-2025-1616109` | 0 | https://www.frontiersin.org/journals/oral-health/articles/10.3389/froh.2025.1616109/full | `10.3389/froh.2025.1616109` | Small CC-BY exemplar photo atlas across DDE continuum. |
| `mdpi-genes-16-822-2025` | 0 | https://www.mdpi.com/2073-4425/16/7/822 | `10.3390/genes16070822` | 24 families; photo-only subset usable. |
| `zenodo-14622450-athletes-caries-erosion` | 0 | https://zenodo.org/records/14622450 | `10.5281/zenodo.14622450|zenodo:14622450` | Tabular only; no photos. usable_subset=none for image pipeline. |

### OPEN / topic = soft_tissue (20)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `dryad-tcm-tongue-1c59zw48r` | 6719 | https://datadryad.org/dataset/doi:10.5061/dryad.1c59zw48r | `10.5061/dryad.1c59zw48r` | Tongue-only; overlap with geographic/fissured tongue classes via cracked/tooth-marked labels. |
| `roboflow-tesisdientes-oral-diseases-4162` | 4162 | https://universe.roboflow.com/tesisdientes/oral-diseases-5ctay-rqpxs | `roboflow:tesisdientes/oral-diseases-5ctay-rqpxs` | Distinct from excluded roboflow-oral-lesion2; class taxonomy needs iter-3 verification |
| `figshare-smartom-31341790` | 2469 | https://doi.org/10.6084/m9.figshare.31341790.v1 | `10.6084/m9.figshare.31341790` | 331 subjects, Android + iOS cameras, real-world clinical lighting; Scientific Data 2026 |
| `scidb-tongue-inquiry-8417299d` | 1194 | https://www.scidb.cn/en/detail?dataSetId=8417299de5ef4f3db5ec62e01a969d54 | `-` | First TCM tongue+inquiry combo; landing 503 at fetch. |
| `github-autooral-2024` | 420 | https://github.com/wurenkai/HF-UNet-and-Autooral-dataset | `10.1038/s41598-024-69125-9|github:wurenkai/HF-UNet-and-Autooral-dataset` | 80 clinical cases, 24-bit RGB 256x256, labeled by 3 dentists; Google Drive mirror; strongest aphthous/herpes/OSCC clinic |
| `mendeley-mhjyrn35p4` | 323 | https://data.mendeley.com/datasets/mhjyrn35p4/2 | `10.17632/mhjyrn35p4.2|mendeley:mhjyrn35p4` | 165 benign + 158 malignant original; augmented folder separate; mobile + intraoral camera |
| `dryad-modid-multispectral-nvx0k6dxw` | 243 | https://datadryad.org/dataset/doi:10.5061/dryad.nvx0k6dxw | `10.5061/dryad.nvx0k6dxw` | Dryad deposition separate from Zenodo mirror; 91 participants, 16 bands 460-600nm |
| `mendeley-ndb-ufes-bbmmm4wgr8-reference-only` | 237 | https://data.mendeley.com/datasets/bbmmm4wgr8/4 | `10.17632/bbmmm4wgr8.4|mendeley:bbmmm4wgr8` | EXCLUDED BY MODALITY BLACKLIST (histology) - logged for LATAM coverage transparency only; UFES Brazil Federal University |
| `mpox-skin-lesion-v2-hfmd-subset` | 161 | https://github.com/mHealthBuet/Mpox-Skin-Lesion-Dataset-v2 | `10.1016/j.bspc.2024.106742|github:mHealthBuet/Mpox-Skin-Lesion-Dataset-v2` | usable_subset = oral/perioral HFMD vesicles only. NC license. |
| `kaggle-zaidpy-oral-cancer-dataset` | 131 | https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset | `kaggle:zaidpy/oral-cancer-dataset` | Clinical photos (NOT histology), 256x256 jpg |
| `roboflow-oral-lesion2` | 39 | https://universe.roboflow.com/kunchidsong-phosri/oral-lesion2 | `roboflow:kunchidsong-phosri/oral-lesion2` | Small dataset (<50 img target); useful as supplemental for detection; flagged for iteration 2 verification |
| `dib-bangladesh-6class` | 0 | https://www.sciencedirect.com/science/article/pii/S2352340924009326 | `10.1016/j.dib.2024.110926` | Multi-source Bangladesh Prescription Point LabAid IbnSina; 232 patients; N_images to confirm; usable_subset intraoral ph |
| `figshare-code-oral-mucosa-30550889` | 0 | https://figshare.com/articles/dataset/_b_CODE_-_b_i_Comprehensive_Oral_mucosa_Database_with_Explanations_i_/30550889 | `10.6084/m9.figshare.30550889|figshare:30550889` | 110 participants, mobile-phone intraoral; lichen planus+lichenoid explicit |
| `github-priyanshu9898-oral-disease-6class` | 0 | https://github.com/Priyanshu9898/Oral-Disease-Classification | `github:Priyanshu9898/Oral-Disease-Classification` | GDrive file 13NFWEqtL_3Vxsr02ehXoYTapp63dMIDM; perio classes = calculus+gingivitis |
| `huggingface-lines-open-domain-oral-disease-qa` | 0 | https://huggingface.co/datasets/Lines/Open-Domain-Oral-Disease-QA-Dataset | `-` | VLM-style; usable_subset=images |
| `kaggle-bavithravairam-mouth-ulcer` | 0 | https://www.kaggle.com/datasets/bavithravairam/mouth-ulcer | `kaggle:bavithravairam/mouth-ulcer` | LOW-CONFIDENCE metadata |
| `kaggle-bavithravairam-oral-ulcer` | 0 | https://www.kaggle.com/datasets/bavithravairam/oral-ulcer | `kaggle:bavithravairam/oral-ulcer` |  |
| `kaggle-muhammadatef-oral-cancer-images-classification` | 0 | https://www.kaggle.com/datasets/muhammadatef/oral-cancer-images-for-classification | `kaggle:muhammadatef/oral-cancer-images-for-classification` | WebFetch 503; possibly histology via ashenafifasilkebede link. Manual verify. |
| `kaggle-shlokmohanty-ulcer-classification` | 0 | https://www.kaggle.com/datasets/shlokmohanty/ulcer-classification | `kaggle:shlokmohanty/ulcer-classification` | RISK: may mix oral+skin ulcers |
| `nature-s41597-024-04099-x-multispectral` | 0 | https://www.nature.com/articles/s41597-024-04099-x | `10.1038/s41597-024-04099-x` | 91 participants (15 healthy + 76 diseased); Dryad-hosted; multispectral but in-vivo clinical, NOT histology. Useful for  |

### OPEN / topic = ortho (5)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `github-omni-2025` | 4166 | https://github.com/RoundFaceJ/OMNI | `github:RoundFaceJ/OMNI` | 5 standard views incl. frontal/lateral/occlusal. Includes brackets, aligners, ortho appliances, malocclusion traits. Mea |
| `teethseg-io150k-rgb08k` | 800 | https://zoubo9034.github.io/TeethSEG/ | `-` | Parent IO150K mixes 80K rendered-from-3D + 70K plaster photos + 800 real intraoral RGB. USABLE_SUBSET = RGB0.8K only (re |
| `mendeley-4mxj6rpv48-facial-profile` | 400 | https://data.mendeley.com/datasets/4mxj6rpv48/2 | `10.17632/4mxj6rpv48.2|mendeley:4mxj6rpv48` | BORDERLINE extraoral profile (not intraoral). Ortho diagnosis context. |
| `roboflow-orthodontic-noybl` | 218 | https://universe.roboflow.com/orthodontic/orthodontic-noybl | `roboflow:orthodontic/orthodontic-noybl` | Small (~218 imgs); needs manual verification that images are intraoral photos (not x-ray). Listed for completeness; low  |
| `aap-oral-health-image-gallery-pediatric` | 50 | https://www.aap.org/en/patient-care/oral-health/oral-health-image-gallery/ | `-` | ~50 imgs estimate. Non-commercial-only. |

### OPEN / topic = pediatric (6)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `mendeley-6zsnhrds9t-noncarious-teeth` | 9562 | https://data.mendeley.com/datasets/6zsnhrds9t/1 | `10.17632/6zsnhrds9t.1|mendeley:6zsnhrds9t` | Pediatric intraoral, explicitly non-carious. Re-label candidate for MIH/hypoplasia/fluorosis. |
| `github-omni-2025` | 4166 | https://github.com/RoundFaceJ/OMNI | `github:RoundFaceJ/OMNI` | 5 standard views incl. frontal/lateral/occlusal. Includes brackets, aligners, ortho appliances, malocclusion traits. Mea |
| `teethseg-io150k-rgb08k` | 800 | https://zoubo9034.github.io/TeethSEG/ | `-` | Parent IO150K mixes 80K rendered-from-3D + 70K plaster photos + 800 real intraoral RGB. USABLE_SUBSET = RGB0.8K only (re |
| `mendeley-4mxj6rpv48-facial-profile` | 400 | https://data.mendeley.com/datasets/4mxj6rpv48/2 | `10.17632/4mxj6rpv48.2|mendeley:4mxj6rpv48` | BORDERLINE extraoral profile (not intraoral). Ortho diagnosis context. |
| `roboflow-orthodontic-noybl` | 218 | https://universe.roboflow.com/orthodontic/orthodontic-noybl | `roboflow:orthodontic/orthodontic-noybl` | Small (~218 imgs); needs manual verification that images are intraoral photos (not x-ray). Listed for completeness; low  |
| `aap-oral-health-image-gallery-pediatric` | 50 | https://www.aap.org/en/patient-care/oral-health/oral-health-image-gallery/ | `-` | ~50 imgs estimate. Non-commercial-only. |

### OPEN / topic = restorative (8)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `roboflow-bitcamp-dental-j1vge` | 5989 | https://universe.roboflow.com/bitcamp/dental-j1vge | `roboflow:bitcamp/dental-j1vge` | modality_mixed:true pending direct verification |
| `roboflow-intra-oral-scanner-vgggs-r4xwu` | 5758 | https://universe.roboflow.com/intraoral-scanner-vgggs/intra-oral-scanner-r4xwu | `roboflow:intraoral-scanner-vgggs/intra-oral-scanner-r4xwu` | Modality inferred from project name; class list unverified (WebFetch 403). Low val acc suggests noisy labels. |
| `alphadent-zftu-v1.1` | 1320 | https://github.com/ZFTurbo/AlphaDent | `github:ZFTurbo/AlphaDent` | DSLR intraoral macro (Canon 6D mk II + 100mm macro + intraoral mirror). Includes wear facets labels — directly in non-ca |
| `roboflow-dental-mate-crown-detection` | 362 | https://universe.roboflow.com/dental-mate/crown-detection-mdfa5 | `roboflow:dental-mate/crown-detection-mdfa5` | Modality mixed — filter to intraoral photos. |
| `roboflow-orthodontic-noybl` | 218 | https://universe.roboflow.com/orthodontic/orthodontic-noybl | `roboflow:orthodontic/orthodontic-noybl` | Small (~218 imgs); needs manual verification that images are intraoral photos (not x-ray). Listed for completeness; low  |
| `roboflow-ai-dentistry-yang-dental` | 0 | https://universe.roboflow.com/ai-in-dentistry/ai-in-dentistry-images-using-intraoral-cam-at-yang-dental | `roboflow:ai-in-dentistry/ai-in-dentistry-images-using-intraoral-cam-at-yang-dental` | Intraoral wand camera. Good for in-situ restoration ctx. |
| `roboflow-havij-caries-tsrca` | 0 | https://universe.roboflow.com/havij/caries-tsrca-guzap | `roboflow:havij/caries-tsrca-guzap` | Modality mixed — intraoral subset must be filtered. Good for pre-training after triage. |
| `roboflow-restoration-filling-aggregate` | 0 | https://universe.roboflow.com/search?q=class%3Afilling+intraoral | `-` | Aggregator placeholder — iter_2 must WebFetch per-project to separate intraoral-photo subsets from X-ray ones. |

### OPEN / topic = surgical (1)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `roboflow-dental-qlzfr-dental-problems` | 0 | https://universe.roboflow.com/dental-qlzfr/dental-problems | `roboflow:dental-qlzfr/dental-problems` | Class pattern strongly suggests panoramic/periapical X-ray — needs manual verification; probable BLACKLIST if radiograph |

### OPEN / topic = view_protocol (5)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `mendeley-6zsnhrds9t` | 9562 | https://data.mendeley.com/datasets/6zsnhrds9t | `mendeley:6zsnhrds9t` | Explicit per-view folders: maxillary/mandibular x front/right/left/occlusal, across 8 subcategories. |
| `figshare-siop-29761292` | 1350 | https://figshare.com/articles/dataset/SIOP_dataset/29761292 | `10.6084/m9.figshare.29761292|figshare:29761292` | 150 SPIP series × 9 dentists = 1350 photos; strict 5-view protocol; quality-graded on 5 dimensions |
| `zenodo-10580117-teledentistry-peru` | 1305 | https://zenodo.org/doi/10.5281/zenodo.10580117 | `10.5281/zenodo.10580117|zenodo:10580117` | 87 patients × 15 intraoral photos (3 per view × 5 views) captured by family members using Xiaomi Redmi 9A + handheld mir |
| `mendeley-3253gj88rr` | 1096 | https://data.mendeley.com/datasets/3253gj88rr | `mendeley:3253gj88rr` | Macro frontal anterior — useful for frontal/macro close-up protocol. |
| `osf-tgm5n-dental-ai-resources-list` | 0 | https://osf.io/tgm5n/ | `10.17605/OSF.IO/TGM5N` | Meta-resource; companion OSF a5pfe+mf897. Mine for un-logged photo entries in iter-4. |

### OPEN / topic = non_western (6)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `arxiv-2511.04948-code` | 50000 | https://arxiv.org/abs/2511.04948 | `10.48550/arXiv.2511.04948|arxiv:2511.04948` | 8775 checkups 4800 patients 2018-2025; also 8056 radiographs excluded from usable subset; usable_subset 50000 intraoral  |
| `segmentanytooth-ump-5000` | 5000 | https://www.sciencedirect.com/science/article/pii/S1991790225000030 | `10.1016/j.jds.2025.01.003` | Dental Public Health Faculty Odonto-Stomatology HCMC University of Medicine and Pharmacy; 953 subjects 32.8%M 67.2%F; 20 |
| `datasetninja-dentalai-2495` | 2495 | https://datasetninja.com/dentalai | `-` | 28904 labeled objects; train/val/test 1991/254/250; 1018 MB Supervisely format; CC BY 4.0; author-region attribution unc |
| `mendeley-ndb-ufes-bbmmm4wgr8-reference-only` | 237 | https://data.mendeley.com/datasets/bbmmm4wgr8/4 | `10.17632/bbmmm4wgr8.4|mendeley:bbmmm4wgr8` | EXCLUDED BY MODALITY BLACKLIST (histology) - logged for LATAM coverage transparency only; UFES Brazil Federal University |
| `dib-bangladesh-6class` | 0 | https://www.sciencedirect.com/science/article/pii/S2352340924009326 | `10.1016/j.dib.2024.110926` | Multi-source Bangladesh Prescription Point LabAid IbnSina; 232 patients; N_images to confirm; usable_subset intraoral ph |
| `nature-sd-osmf-oscc-2024` | 0 | https://www.nature.com/articles/s41597-024-03836-6 | `10.1038/s41597-024-03836-6` | ORCHID-related Indian OSMF/OSCC intraoral image dataset high-resolution; exact count to confirm |

## REQUEST — Gated (email / DUA / institutional access) (65 entries)

### REQUEST / topic = caries (12)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `sunyatsen-zengcheng-pediatric-7671-2026` | 7671 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12905764/ | `PMC12905764` | 3913 occlusal + 3758 smooth from 524 kids; custom retractor/mirror/LED device; questionnaires linked. |
| `bmc-hibogi-cimahi-indonesia-3221-2025` | 3221 | https://doi.org/10.1186/s12903-025-07486-x | `10.1186/s12903-025-07486-x` | Cimahi; 5 schools + 3 community health centers; 3221 JPG; train 2266/val 635/test 320. First SEA primary-school caries d |
| `bmc-mixed-dentition-intraoral-2025-06866` | 2467 | https://link.springer.com/article/10.1186/s12903-025-06866-7 | `10.1186/s12903-025-06866-7` | 5-view protocol with MIH/fluorosis/hypoplasia as explicit classes. |
| `mdpi-diagnostics-ksu-435` | 435 | https://www.mdpi.com/2075-4418/16/6/862 | `10.3390/diagnostics16060862` | King Saud University Riyadh; pediatric oral photographs; patient-level stratified splitting; ResNet-18 MobileNetV3 Effic |
| `aihub-korea-71509-dental-intraoral-clinical` | 0 | https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71509 | `-` | Korean-government AI training dataset from SNU Dental Hospital. 6 clinical views per patient. Critical Asian-region sour |
| `github-dlcariesscreen-ucla` | 0 | https://github.com/liangyuandg/DLCariesScreen | `github:liangyuandg/DLCariesScreen` | Dataset NOT public due to privacy/commercial concerns; request via liangyuandg@ucla.edu or Prof. Leiying Miao. Logged wi |
| `github-tooth-segmentation44-yolov8` | 0 | https://github.com/YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44 | `github:YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44` | Chinese-authored; distinct from sister repo rvg-v1301 which is X-ray film; tooth-segmentation44 appears intraoral photo  |
| `jmir-e91239-colorado-fluorosis-2026` | 0 | https://www.researchprotocols.org/2026/1/e91239 | `10.2196/91239` | Protocol Jan 2026. ~300/1000 collected. Smartphone + intraoral camera. Not yet public. |
| `mdpi-jcm-14-8959-enamel-caries-dl-2025` | 0 | https://www.mdpi.com/2077-0383/14/24/8959 | `10.3390/jcm14248959` | Hypoplasia+fluorosis as distractors; data-on-request. |
| `nyu-caredaway-sdf-school-ny-10620` | 0 | https://datacatalog.med.nyu.edu/dataset/10620 | `-` | 7418 children, 4100 completers. First US large-scale pediatric SDF cohort; photos uncertain — contact PI Ryan Richard Ru |
| `sechenov-ai-214-pediatric-dental-morbidity-moscow` | 0 | https://ai.sechenov.ru/datasets/214 | `-` | Photo modality NOT confirmed — may be structured clinical records. Verify via direct request. |
| `sfedu-yufu-caries-smartphone-2026` | 0 | https://sfedu.ru/press-center/news/80063 | `-` | Denis Krivoguz team at Southern Federal University (Rostov-on-Don). Built YOLOv8 caries detector on smartphone camera in |

### REQUEST / topic = perio (9)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `nanchang-university-kouqiang-3869-perio-screening-2025-lead` | 3869 | http://manu45.magtech.com.cn/Jwk_kqyxyj/CN/abstract/abstract2506.shtml | `-` | 578 subjects. OUTREACH lead iter-5. |
| `fdtooth-scidata-2025` | 1800 | https://physionet.org/content/fdtooth/1.0.0/ | `-` | Usable subset = intraoral photos with recession labels; CBCT part blacklisted. |
| `bmc-s12903-024-04460-x-chronic-gingivitis-683` | 683 | https://doi.org/10.1186/s12903-024-04460-x | `10.1186/s12903-024-04460-x` | Li 2024 BMC OH 24:814; 134 volunteers |
| `moharrami-ragadio-2024-gingivitis-666-kaggle-lead` | 666 | https://onlinelibrary.wiley.com/doi/full/10.1111/cdoe.70001 | `-` | Cited as Kaggle dataset; exact slug not located iter-3. |
| `chau-hku-gingivitis-567-2023` | 567 | https://doi.org/10.1016/j.identj.2023.03.007 | `10.1016/j.identj.2023.03.007|PMC12627268` | Originating paper for 567-img set cited in PMC12627268 SR; HKU RGC grant UGC/FDS13/E01/22 |
| `mdpi-diagnostics-plaque-oleary` | 531 | https://www.mdpi.com/journal/diagnostics | `-` | Smartphone-based plaque scoring. |
| `terven-ipn-plaque-531-oleary-2025` | 531 | https://www.mdpi.com/2075-4418/15/2/231 | `10.3390/diagnostics15020231` | 177 individuals, multi-smartphone; distinct from mdpi-diagnostics-plaque-oleary |
| `sechenov-ai-214-pediatric-dental-morbidity-moscow` | 0 | https://ai.sechenov.ru/datasets/214 | `-` | Photo modality NOT confirmed — may be structured clinical records. Verify via direct request. |
| `ssrn-6281532-airc-labden-plaque-ortho` | 0 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6281532 | `-` | Ortho-perio crossover; contact Chi Nguyen Anh et al. |

### REQUEST / topic = noncarious (18)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `mih-cnn-springer-2022-3241` | 3241 | https://link.springer.com/article/10.1007/s00784-022-04552-4 | `10.1007/s00784-022-04552-4` | 3241 intraoral photos (train 2596 / test 649). Contact authors iter 2. |
| `bmc-mixed-dentition-intraoral-2025-06866` | 2467 | https://link.springer.com/article/10.1186/s12903-025-06866-7 | `10.1186/s12903-025-06866-7` | 5-view protocol with MIH/fluorosis/hypoplasia as explicit classes. |
| `human-tooth-crack-nir-2024` | 1194 | https://link.springer.com/article/10.1007/s10439-024-03615-9 | `10.1007/s10439-024-03615-9` | 593 cracked + 601 non-cracked from NIR videos. Borderline-photo modality. Enamel-crack niche. |
| `pku-tooth-wear-grading-388` | 388 | https://www.sciencedirect.com/science/article/pii/S1991790224001594 | `10.1016/j.jds.2024.04.022` | In-zone tooth wear grading. F1=0.89. Request-based. |
| `abha-ksa-mih-520-kingkhalid-2026` | 0 | https://pubmed.ncbi.nlm.nih.gov/41746461/ | `-` | n=520 Saudi children Feb-2026; outreach for photo-level release. |
| `cleft-mih-6432-coi-2025-06311` | 0 | https://link.springer.com/article/10.1007/s00784-025-06311-7 | `10.1007/s00784-025-06311-7` | 6432 teeth from 290 cleft patients via 40-inch screen photo review. |
| `coi-skin-dde-pediatric-2025-06326` | 0 | https://pubmed.ncbi.nlm.nih.gov/40227455/ | `10.1007/s00784-025-06326-0` | Genodermatoses cohort, rare-disease AI enrichment candidate. |
| `dongthap-vietnam-mih-2025-40589688` | 0 | https://pubmed.ncbi.nlm.nih.gov/40589688/ | `-` | Southern Vietnam counterpart to Northern 1834 study; outreach iter_5. |
| `frontiers-mih-central-china-1568` | 0 | https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1088703/full | `10.3389/fphys.2023.1088703` | 1568 with MIH+DF co-labels in endemic fluorosis area; photos not released; authors contactable. |
| `github-yunwu2024-dental-fluorosis` | 0 | https://github.com/yunwu2024/dental_fluorosis | `github:yunwu2024/dental_fluorosis` | Placeholder repo; release pending paper acceptance. Monitor iter 2-3. |
| `heliyon-tooth-crack-deeplabv3plus-2024` | 0 | https://www.cell.com/heliyon/fulltext/S2405-8440(24)01923-6 | `-` | Craze-line / enamel-crack niche; images on request. |
| `integrated-fluorosis-grading-bspc-2024` | 0 | https://www.sciencedirect.com/science/article/abs/pii/S1746809424005688 | `10.1016/j.bspc.2024.106829` | U-Net tooth seg + CNN+Transformer classifier. |
| `jmir-e91239-colorado-fluorosis-2026` | 0 | https://www.researchprotocols.org/2026/1/e91239 | `10.2196/91239` | Protocol Jan 2026. ~300/1000 collected. Smartphone + intraoral camera. Not yet public. |
| `mdpi-jcm-14-8959-enamel-caries-dl-2025` | 0 | https://www.mdpi.com/2077-0383/14/24/8959 | `10.3390/jcm14248959` | Hypoplasia+fluorosis as distractors; data-on-request. |
| `mdpi-mih-oct-photonics-799-2025` | 0 | https://www.mdpi.com/2304-6732/12/8/799 | `10.3390/photonics12080799` | In-vivo OCT paired with clinical photos. |
| `plos-one-0310420-mexican-fluorosis-mih-573` | 0 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310420 | `10.1371/journal.pone.0310420` | 573 children, dual-endemic fluorosis+MIH phenotype; outreach iter_5. |
| `preprints-2025-tooth-wear-portuguese-community` | 0 | https://www.preprints.org/frontend/manuscript/11e1eb8379393f65e572e1329eef56ab/download_pub | `-` | Attrition+NCCL 12.3%/9%. |
| `s41598-025-25960-y-vietnam-mih-1834-northern` | 0 | https://www.nature.com/articles/s41598-025-25960-y | `10.1038/s41598-025-25960-y` | n=1834, MIH prev 12.7%; photos on request. |

### REQUEST / topic = soft_tissue (11)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `memosa-dataset-2026` | 30039 | https://www.nature.com/articles/s41597-026-06998-7 | `-` | LARGEST soft-tissue photo dataset found. Mobile device cameras. Biopsy verification where applicable. Workbench-gated ac |
| `memosa-multi-country-oral` | 30039 | https://www.memosa.org/ | `-` | Largest smartphone intraoral mucosa archive; 5-country collection. |
| `bdj-cairo-9201` | 9201 | https://www.nature.com/articles/s41415-025-9007-6 | `10.1038/s41415-025-9007-6` | Oral Medicine Clinic Faculty of Dentistry Cairo University; 4405 normal 2314 low 2482 high; labeled by 2 oral medicine s |
| `zenodo-14571990` | 9201 | https://zenodo.org/records/14571990 | `10.5281/zenodo.14571990|zenodo:14571990` | Mobile + DSLR; 4405 normal / 2314 low-risk / 2482 high-risk; labelled by oral medicine specialists |
| `zenodo-10664056` | 3000 | https://doi.org/10.5281/zenodo.10664056 | `10.5281/zenodo.10664056` | Mobile phone photos; patient-wise CSV with smoking/betel/alcohol risk factors; restricted-access (institutional email re |
| `aikosh-india-oral-cancer` | 0 | https://aikosh.indiaai.gov.in/home/datasets/details/oral_cancer_imaging_and_clinical_dataset.html | `-` | IndiaAI AIKosh platform; requires account/request; confirm image count and modality split |
| `arxiv-2511-21582-multimodal-oral-lesions-16class` | 0 | https://arxiv.org/abs/2511.21582 | `arxiv:2511.21582` | Nov 2025; possibly reuses CLASEG-2072 given 16-class overlap; verify iter-5 |
| `github-tooth-segmentation44-yolov8` | 0 | https://github.com/YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44 | `github:YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44` | Chinese-authored; distinct from sister repo rvg-v1301 which is X-ray film; tooth-segmentation44 appears intraoral photo  |
| `opmdcare-atlas-photos-2026` | 0 | https://opmdcare.com/atlas-photos/ | `-` | Educational atlas; no DOI; author-contact (Bordeaux lead). |
| `oralgpt-mmoral-dfull-dpartial-arxiv-2510-13911` | 0 | https://arxiv.org/abs/2510.13911 | `arxiv:2510.13911` | iter-3 resolution: CONFIRMED PRIVATE. Only DPublic+MMOral-OPG-Bench (panoramic, out-of-scope) on HF. Contact isjinghao@g |
| `tongue-lesions-bmc-623patients` | 0 | https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-024-01234-3 | `10.1186/s12880-024-01234-3` | Geographic tongue explicit; on-request |

### REQUEST / topic = ortho (7)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `peerj-20140-bengbu-occlusion-7200` | 7200 | https://peerj.com/articles/20140/ | `10.7717/peerj.20140|peerj:20140` | 5000@45° + 2200@90°, 1920x1280. Dataset institutional, code only. |
| `mdpi-dentistry-malocclusion-5854` | 5854 | https://www.mdpi.com/2304-6767/14/1/60 | `10.3390/dj14010060` | 5 standard views labeled with 17 IOTN classes; YOLOv11 benchmark; origin to confirm China authorship noted |
| `fdtooth-physionet-hku-2025` | 241 | https://physionet.org/content/fdtooth/1.0.0/ | `10.13026/v9xk-dy61` | Photo-subset usable (MODALITY_MIXED=true, reject CBCT half). Explicit bracket-present/absent labels. |
| `schwarzmaier-143-ecc-anterior-deciduous-2024` | 143 | https://www.mdpi.com/2077-0383/13/17/5215 | `10.3390/jcm13175215` | 107 ECC + 36 control. High-res 2784x1856 macro-flash. Request corresponding author. |
| `nyu-caredaway-sdf-school-ny-10620` | 0 | https://datacatalog.med.nyu.edu/dataset/10620 | `-` | 7418 children, 4100 completers. First US large-scale pediatric SDF cohort; photos uncertain — contact PI Ryan Richard Ru |
| `sechenov-ai-214-pediatric-dental-morbidity-moscow` | 0 | https://ai.sechenov.ru/datasets/214 | `-` | Photo modality NOT confirmed — may be structured clinical records. Verify via direct request. |
| `ssrn-6281532-airc-labden-plaque-ortho` | 0 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6281532 | `-` | Ortho-perio crossover; contact Chi Nguyen Anh et al. |

### REQUEST / topic = pediatric (11)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `peerj-20140-bengbu-occlusion-7200` | 7200 | https://peerj.com/articles/20140/ | `10.7717/peerj.20140|peerj:20140` | 5000@45° + 2200@90°, 1920x1280. Dataset institutional, code only. |
| `bmc-hibogi-cimahi-indonesia-3221-2025` | 3221 | https://doi.org/10.1186/s12903-025-07486-x | `10.1186/s12903-025-07486-x` | Cimahi; 5 schools + 3 community health centers; 3221 JPG; train 2266/val 635/test 320. First SEA primary-school caries d |
| `bmc-tbilisi-humanitarian-2864-georgia-2025` | 2864 | https://doi.org/10.1186/s12903-025-06500-6 | `10.1186/s12903-025-06500-6` | Tbilisi Humanitarian Univ; 358 children, 8 standard projections, smartphone. First Georgia/Caucasus dataset. |
| `fdtooth-physionet-hku-2025` | 241 | https://physionet.org/content/fdtooth/1.0.0/ | `10.13026/v9xk-dy61` | Photo-subset usable (MODALITY_MIXED=true, reject CBCT half). Explicit bracket-present/absent labels. |
| `schwarzmaier-143-ecc-anterior-deciduous-2024` | 143 | https://www.mdpi.com/2077-0383/13/17/5215 | `10.3390/jcm13175215` | 107 ECC + 36 control. High-res 2784x1856 macro-flash. Request corresponding author. |
| `abha-ksa-mih-520-kingkhalid-2026` | 0 | https://pubmed.ncbi.nlm.nih.gov/41746461/ | `-` | n=520 Saudi children Feb-2026; outreach for photo-level release. |
| `dongthap-vietnam-mih-2025-40589688` | 0 | https://pubmed.ncbi.nlm.nih.gov/40589688/ | `-` | Southern Vietnam counterpart to Northern 1834 study; outreach iter_5. |
| `frontiers-mih-central-china-1568` | 0 | https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1088703/full | `10.3389/fphys.2023.1088703` | 1568 with MIH+DF co-labels in endemic fluorosis area; photos not released; authors contactable. |
| `nyu-caredaway-sdf-school-ny-10620` | 0 | https://datacatalog.med.nyu.edu/dataset/10620 | `-` | 7418 children, 4100 completers. First US large-scale pediatric SDF cohort; photos uncertain — contact PI Ryan Richard Ru |
| `plos-one-0310420-mexican-fluorosis-mih-573` | 0 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310420 | `10.1371/journal.pone.0310420` | 573 children, dual-endemic fluorosis+MIH phenotype; outreach iter_5. |
| `s41598-025-25960-y-vietnam-mih-1834-northern` | 0 | https://www.nature.com/articles/s41598-025-25960-y | `10.1038/s41598-025-25960-y` | n=1834, MIH prev 12.7%; photos on request. |

### REQUEST / topic = restorative (5)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `oii-ds-benchmark-wang-2023` | 19074 | https://doi.org/10.1016/j.compbiomed.2023.107620 | `10.1016/j.compbiomed.2023.107620` | Subset-A 3834 CT imgs BLACKLISTED. Subset-B 15240 intraoral implant photos = restorative scope. usable_subset:Subset-B.  |
| `osaka-prosthesis-1904-takahashi-2021` | 1904 | https://www.nature.com/articles/s41598-021-81202-x | `10.1038/s41598-021-81202-x` | Not publicly hosted — contact Osaka University Dental Hospital, Dept of Prosthodontics. |
| `osaka-prosthodontics-contact-draft-2026` | 1904 | https://global.dent.osaka-u.ac.jp/ | `-` | Parent osaka-prosthesis-1904-takahashi-2021 excluded in iter-1. Contacts drafted toshi-t@dent.osaka-u.ac.jp, mameno.tomo |
| `aihub-kr-restorative-probe-2026` | 0 | https://www.aihub.or.kr/aihubdata/data/list.do | `-` | Catalogue confirmed; individual dataset IDs require Korean login. Needs KR-resident collaborator. |
| `github-tooth-segmentation44-yolov8` | 0 | https://github.com/YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44 | `github:YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44` | Chinese-authored; distinct from sister repo rvg-v1301 which is X-ray film; tooth-segmentation44 appears intraoral photo  |

### REQUEST / topic = view_protocol (8)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `memosa-multi-country-oral` | 30039 | https://www.memosa.org/ | `-` | Largest smartphone intraoral mucosa archive; 5-country collection. |
| `sunyatsen-zengcheng-pediatric-7671-2026` | 7671 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12905764/ | `PMC12905764` | 3913 occlusal + 3758 smooth from 524 kids; custom retractor/mirror/LED device; questionnaires linked. |
| `bengbu-7200-occlusion-classification-2025` | 7200 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12478310/ | `PMC12478310` | 7200 photos / 6100 patients Jun22-Jun23; only code released. |
| `charite-berlin-5266-angle-lateral-2025` | 5266 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12861064/ | `10.1186/s12903-025-07550-6|PMC12861064` | Largest single-view lateral intraoral orthodontic corpus identified. |
| `segmentanytooth-hcmc` | 5000 | https://github.com/thangngoc89/SegmentAnyTooth | `github:thangngoc89/SegmentAnyTooth` | 1000 subjects x 5 standard views (953 subjects usable). Weights/data on request. |
| `jocpd-2026-011-pediatric-5view` | 3100 | https://www.jocpd.com/articles/10.22514/jocpd.2026.011 | `10.22514/jocpd.2026.011` | 620 pediatric patients × 5 views = 3100 photos with explicit view-type labels; Inception-ResNet-v2 + Faster R-CNN benchm |
| `smart-om-smartphone-oral` | 2469 | https://github.com/search?q=SMART-OM+smartphone+oral | `-` | Smartphone iOS+Android capture, 331 subjects — camera-protocol axis. |
| `teethdreamer-5photo-protocol` | 0 | https://arxiv.org/abs/2407.11419 | `arxiv:2407.11419` | Usable subset = 5 photos per subject. Meshes are blacklisted by our modality filter. |

### REQUEST / topic = ru_cis (6)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `alphadent-kaggle-competition` | 1320 | https://www.kaggle.com/competitions/alpha-dent/data | `kaggle:alpha-dent/data` | Kaggle mirror of AlphaDent. Train/val split by patient: 273 train / 22 val. Hidden test set for leaderboard. Task = inst |
| `sechenov-ai-205-adult-dental-photos-polygon` | 0 | https://ai.sechenov.ru/datasets/205 | `-` | 300 adult patients per portal listing; distinct from iter-1/2 Sechenov entries (Sjogren/DataMed/platform-lead). |
| `sechenov-ai-214-pediatric-dental-morbidity-moscow` | 0 | https://ai.sechenov.ru/datasets/214 | `-` | Photo modality NOT confirmed — may be structured clinical records. Verify via direct request. |
| `sechenov-datamed-ai-platform-lead` | 0 | https://ai.sechenov.ru/datasets | `-` | Platform hosts 26 datasets as of 2026-04; only dental = already-excluded Sjögren histology. Outreach lead. |
| `sechenov-minzdrav-sjogren-salivary-2024` | 0 | https://www.sechenov.ru/pressroom/news/vklad-v-delo-tsifrovizatsii-sechenovskiy-universitet-sozdal-i-vylozhil-na-platformu-minzdrava-dlya-o/ | `-` | OUT-OF-SCOPE (histology, not intraoral photo) but logged as negative anchor + institutional lead. Institute of Clinical  |
| `sfedu-yufu-caries-smartphone-2026` | 0 | https://sfedu.ru/press-center/news/80063 | `-` | Denis Krivoguz team at Southern Federal University (Rostov-on-Don). Built YOLOv8 caries detector on smartphone camera in |

### REQUEST / topic = non_western (8)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `bdj-cairo-9201` | 9201 | https://www.nature.com/articles/s41415-025-9007-6 | `10.1038/s41415-025-9007-6` | Oral Medicine Clinic Faculty of Dentistry Cairo University; 4405 normal 2314 low 2482 high; labeled by 2 oral medicine s |
| `mdpi-dentistry-malocclusion-5854` | 5854 | https://www.mdpi.com/2304-6767/14/1/60 | `10.3390/dj14010060` | 5 standard views labeled with 17 IOTN classes; YOLOv11 benchmark; origin to confirm China authorship noted |
| `bmc-hibogi-cimahi-indonesia-3221-2025` | 3221 | https://doi.org/10.1186/s12903-025-07486-x | `10.1186/s12903-025-07486-x` | Cimahi; 5 schools + 3 community health centers; 3221 JPG; train 2266/val 635/test 320. First SEA primary-school caries d |
| `bmc-shahid-beheshti-3215` | 3215 | https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-025-05803-y | `10.1186/s12903-025-05803-y` | Orthodontics Dept Dental School Shahid Beheshti University of Medical Sciences Tehran; professional camera post-cleaning |
| `bmc-tbilisi-humanitarian-2864-georgia-2025` | 2864 | https://doi.org/10.1186/s12903-025-06500-6 | `10.1186/s12903-025-06500-6` | Tbilisi Humanitarian Univ; 358 children, 8 standard projections, smartphone. First Georgia/Caucasus dataset. |
| `mdpi-diagnostics-ksu-435` | 435 | https://www.mdpi.com/2075-4418/16/6/862 | `10.3390/diagnostics16060862` | King Saud University Riyadh; pediatric oral photographs; patient-level stratified splitting; ResNet-18 MobileNetV3 Effic |
| `aikosh-india-oral-cancer` | 0 | https://aikosh.indiaai.gov.in/home/datasets/details/oral_cancer_imaging_and_clinical_dataset.html | `-` | IndiaAI AIKosh platform; requires account/request; confirm image count and modality split |
| `github-tooth-segmentation44-yolov8` | 0 | https://github.com/YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44 | `github:YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44` | Chinese-authored; distinct from sister repo rvg-v1301 which is X-ray film; tooth-segmentation44 appears intraoral photo  |

## UNCLEAR — Ambiguous availability (9 entries)

### UNCLEAR / topic = perio (3)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `iter4-perio-saturation-flag` | 0 |  | `-` | Recommend paused_agents += dental-search-perio for iter-5. Remaining value = outreach (Nanchang 3869, Roboflow slug reso |
| `iter4-surgical-pmc-figure-harvest-plan` | 0 |  | `-` | Separate figure-harvest pipeline needed (not keyword search). Plausible ~50-200 photos per condition across CC-BY PMC ca |
| `iter4-surgical-zone-gap-structural` | 0 |  | `PMC10049628` | 15+ queries en+ru+zh+pt+es+tr yield: (a) DentalTraumaGuide Copenhagen/IADT paywalled; (b) Andreasen Color Atlas copyrigh |

### UNCLEAR / topic = soft_tissue (2)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `roboflow-oralleukoplakia-418` | 418 | https://universe.roboflow.com/search?q=OralLeukoplakia | `-` | Cited in OralGPT DPublic contributing 418 OLK images. |
| `roboflow-ulcers-sw4n4` | 0 | https://universe.roboflow.com/ulcer-detection-sw4n4/ulcers | `roboflow:ulcer-detection-sw4n4/ulcers` | Cited as the Roboflow Ulcer Dataset in Sci Reports 2025 aphthous CNN paper. |

### UNCLEAR / topic = ortho (1)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `osf-tf5r8-data-share-ortho` | 0 | https://osf.io/tf5r8/ | `-` | Project exists but 503 in iter-2 AND iter-3. Needs api.osf.io/v2/nodes/tf5r8/ direct call. |

### UNCLEAR / topic = pediatric (1)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `osf-tf5r8-data-share-ortho` | 0 | https://osf.io/tf5r8/ | `-` | Project exists but 503 in iter-2 AND iter-3. Needs api.osf.io/v2/nodes/tf5r8/ direct call. |

### UNCLEAR / topic = restorative (5)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `khurshid-bmc-2025-panoramic-prosthesis-out-of-scope-note` | 2235 | https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-025-07138-0 | `10.1186/s12903-025-07138-0` | Rejected: panoramic radiograph, not intraoral photo. Logged to prevent re-investigation. |
| `alphadent-v1.2-release-check-negative-2026` | 0 | https://github.com/ZFTurbo/AlphaDent | `github:ZFTurbo/AlphaDent` | Latest tag still v1.1 (2025-07-29). No v1.2 as of 2026-04-22. |
| `hf-mendeley-roboflow-api-probe-2026` | 0 |  | `-` | HF /api/datasets?search=prosthesis/veneer/crown → no new photo datasets. Mendeley API DNS blocked. Roboflow 503. No new  |
| `iter4-surgical-pmc-figure-harvest-plan` | 0 |  | `-` | Separate figure-harvest pipeline needed (not keyword search). Plausible ~50-200 photos per condition across CC-BY PMC ca |
| `iter4-surgical-zone-gap-structural` | 0 |  | `PMC10049628` | 15+ queries en+ru+zh+pt+es+tr yield: (a) DentalTraumaGuide Copenhagen/IADT paywalled; (b) Andreasen Color Atlas copyrigh |

### UNCLEAR / topic = surgical (2)

| ID | n | URL | aliases | notes |
|---|---:|---|---|---|
| `iter4-surgical-pmc-figure-harvest-plan` | 0 |  | `-` | Separate figure-harvest pipeline needed (not keyword search). Plausible ~50-200 photos per condition across CC-BY PMC ca |
| `iter4-surgical-zone-gap-structural` | 0 |  | `PMC10049628` | 15+ queries en+ru+zh+pt+es+tr yield: (a) DentalTraumaGuide Copenhagen/IADT paywalled; (b) Andreasen Color Atlas copyrigh |

## GAP_NOTE — Confirmed dead-ends (do not re-search these) (27 entries)

| ID | URL | notes |
|---|---|---|
| `iter3-gap-note-systematic-review-s41746-01818-5-datasets` | https://www.nature.com/articles/s41746-025-01818-5 | Dataset 50 (Majumdar/Jaykar, 43 TIFF) < 50-floor. Dataset 65 (Shehab 6845 JPG Apache 2.0) — URL in Supp S1 not yet resolved. Iter_4 fetch Supp S1. |
| `memosa-workbench-access-note` | https://workbench.memosa.my | Bulk download NOT offered; registration+platform analytics only. Cannot ingest locally without DUA |
| `iter3-gap-note-dentex-hhs9g-kaggle-caries-modality-unclear` | https://universe.roboflow.com/dentex-hhs9g/kaggle-caries/dataset/2 | Workspace 'dentex' = DENTEX MICCAI23 panoramic X-ray team; 'kaggle-caries' ambiguous. 503 twice. Likely X-ray reupload (blacklisted). Flagged for iter_4 direct verification. |
| `multiclass-oral-mucosa-3246-1013-2025-contact-note` | https://www.sciencedirect.com/science/article/pii/S1746809425008481 | Largest fine-grained mucosa taxonomy seen; iter_4 priority author-contact. |
| `claseg-oral-lesions-2025-contact-note` | https://www.nature.com/articles/s41598-025-03268-1 | Ideal class taxonomy for mucosa remit; no repo DOI. |
| `outreach-kim-2025-plaque-quigley-hein-1394` | https://pmc.ncbi.nlm.nih.gov/articles/PMC12220555/ | OUTREACH Jeong-Hwan Kim; 1094 train + 300 eval. |
| `oralgpt-mucosa-xian-jiaotong-contact-note` | https://arxiv.org/html/2510.13911 | NeurIPS 2025. DFull 480 + DPartial 659. Also DPublic 1280 from 7 public sources. |
| `outreach-wen-2024-gingivitis-826-sci-rep` | https://www.nature.com/articles/s41598-024-70311-y | OUTREACH; data-availability unverified (503). |
| `outreach-nantakeeratipat-2024-plaque-600-thailand` | https://pmc.ncbi.nlm.nih.gov/articles/PMC11797812/ | OUTREACH teerachate@g.swu.ac.th. |
| `outreach-lee-nycu-2025-keratinized-gingiva-576` | https://pmc.ncbi.nlm.nih.gov/articles/PMC12176457/ | OUTREACH sylee@nycu.edu.tw; mucogingival AI unique. |
| `outreach-vaughan-2025-multi-view-35-perio` | https://pmc.ncbi.nlm.nih.gov/articles/PMC12274314/ | OUTREACH; rare 5-view perio, small n=35. |
| `outreach-di-gianfilippo-2025-recession-34-contact` | https://pmc.ncbi.nlm.nih.gov/articles/PMC12123396/ | OUTREACH rdgianfi@umich.edu; first explicit recession labelled photo set. |
| `iter4-surgical-outreach-lead-polish-pediatric-cellulitis` | https://pmc.ncbi.nlm.nih.gov/articles/PMC10049628/ | 27 patients; photos in paper figures (CC-BY), not compiled dataset. Outreach Jedrzejewska et al. for compiled release. |
| `ding-zhejiang-secondary-caries-outreach-2026` | - | Mobile-phone oral photos, YOLOv3 caries; asked for marginal-integrity sub-annotation. |
| `iter2-restorative-gap-note-secondary-caries` | - | Meta entry. No dedicated dataset found. Iter-3 author-contact to Ding (Zhejiang), Khurshid (BMC 2025). |
| `iter3-gap-note-ortho-pediatric` | - | Roboflow bracket/aligner -> only mdpi-5854 & ortho-noybl (excluded). OSF tf5r8/a5pfe/k7zun 503. AI Hub Korea no ortho. ScienceDB/CSTR only 3D STL. AICaries 100k ended. FDTooth v2 none. SDF/Hall/natal  |
| `iter3-perio-recession-gap-note-minimal` | https://pmc.ncbi.nlm.nih.gov/articles/PMC12123396/ | Only Di Gianfilippo 34 on request. Deprioritize recession search iter-4. |
| `iter3-restorative-saturation-report` | - | 0 new fully-public restorative-intraoral datasets. Wang 2025 mixed-dentition (fillings) strongest lead but request-only. Recommend stop_threshold reassessment. |
| `iter4-gap-note-benign-inflammatory-no-dedicated-sets` | - | Confirmed gap after Figshare+ResearchSquare+PMC. Classes exist only inside CLASEG-2072 / BSPC-3246. Iter-5 PMC CC-BY figure harvest: PMC11909381 OLP review, PMC7864300, PMC11649382, PMC11162524. |
| `iter4-gap-note-betel-khat-chromogenic-no-open-photos` | https://pmc.ncbi.nlm.nih.gov/articles/PMC11725193/ | 2025 BMC scoping review 21 studies + Shanghai BTS 250 preschool — no images open. Outreach: Sanaa Dental, NTU Taiwan, Shanghai 9th. |
| `iter4-gap-note-endemic-belts-no-public-photos` | https://pubmed.ncbi.nlm.nih.gov/40741788/ | India meta prev 34.5%; Ethiopia Rift 80%+; outreach: AIIMS Jodhpur, AAU, NUIC Durango, UNC-Cordoba. |
| `iter4-gap-note-strasbourg-d4phenodent` | https://www.phenodent.org/ | 221 genotyped AI persons, 111 families. Outreach: Agnes Bloch-Zupan. |
| `iter4-perio-china-cn-sources-no-new` | https://opendatalab.com | CN probes yield only aggregator blogs pointing to excluded Roboflow sets + Nanchang lead. Saturation. |
| `iter4-perio-gingiva-covid-gap` | https://onlinelibrary.wiley.com/doi/10.1002/rmv.70057 | Avais 2025 SR (107 studies): only case photos embedded in papers. |
| `iter4-perio-mucogingival-graft-donor-gap` | https://www.mdpi.com/2079-4983/15/12/360 | Beyond Lee NYCU 576 (excluded) no open set. Outreach candidate. |
| `iter4-perio-peri-implantitis-photo-zero` | https://www.frontiersin.org/journals/dental-medicine/articles/10.3389/fdmed.2025.1722375/full | Literature ~100% radiograph/CBCT. Coordinate with agent_10 surgical. |
| `iter4-surgical-outreach-lead-iadt-dentaltraumaguide` | https://dentaltraumaguide.org/ | Largest trauma photo atlas worldwide; paywalled. Outreach IADT for de-identified CC-BY research subset. |

---

## FLAT EXCLUDE_IDS (for programmatic match)

```
zenodo-14827784
mendeley-9jnf2jvghy
arxiv-2507.22512-alphadent
mendeley-3253gj88rr
oral-mamba-liu-2024
alphadent-2025
roboflow-gingivitis-fdams
roboflow-gingivitis-t98xc
kaggle-salmansajid05-oral-diseases
github-pknu-calculus
fdtooth-scidata-2025
mdpi-diagnostics-plaque-oleary
alphadent-zftu-v1.1
pku-tooth-wear-grading-388
mendeley-6zsnhrds9t-noncarious-teeth
mih-cnn-springer-2022-3241
github-yunwu2024-dental-fluorosis
figshare-19641750-fluorosis-raw
zenodo-14622450-athletes-caries-erosion
github-autooral-2024
zenodo-10664056
zenodo-14571990
figshare-smartom-31341790
memosa-dataset-2026
mendeley-mhjyrn35p4
roboflow-oral-lesion2
nature-s41597-024-04099-x-multispectral
osaka-prosthesis-1904-takahashi-2021
roboflow-havij-caries-tsrca
roboflow-dental-mate-crown-detection
roboflow-ai-dentistry-yang-dental
roboflow-restoration-filling-aggregate
github-omni-2025
teethseg-io150k-rgb08k
roboflow-orthodontic-noybl
segmentanytooth-hcmc
mendeley-6zsnhrds9t
smart-om-smartphone-oral
memosa-multi-country-oral
teethdreamer-5photo-protocol
alphadent-kaggle-competition
sfedu-yufu-caries-smartphone-2026
sechenov-minzdrav-sjogren-salivary-2024
bdj-cairo-9201
arxiv-2511.04948-code
dib-bangladesh-6class
bmc-shahid-beheshti-3215
mdpi-diagnostics-ksu-435
segmentanytooth-ump-5000
mdpi-dentistry-malocclusion-5854
aikosh-india-oral-cancer
nature-sd-osmf-oscc-2024
mendeley-phtw6rmwzd-enamel-caries
roboflow-project-group13-dl-caries
roboflow-jesse-perring-caries-detection
roboflow-bscs-8th-tooth-cavity-detection
roboflow-yolov8-training-tooth-caries-3
roboflow-jb-okcsk-kaggle-dental-caries
kaggle-maazmakhdoom-dental-cavity
github-dlcariesscreen-ucla
bmc-s12903-024-04460-x-chronic-gingivitis-683
gdrive-nanjing-oral-endoscopy-3365-mirror
chau-hku-gingivitis-567-2023
terven-ipn-plaque-531-oleary-2025
roboflow-tooth-ytblb-dental-plaque-v6
github-priyanshu9898-oral-disease-6class
roboflow-oral-disease-detection-teeth-disease-m1uob-621
roboflow-dental-anomaly-6557
jmir-e91239-colorado-fluorosis-2026
frontiers-mih-central-china-1568
human-tooth-crack-nir-2024
heliyon-tooth-crack-deeplabv3plus-2024
figshare-code-oral-mucosa-30550889
dryad-modid-multispectral-nvx0k6dxw
roboflow-tesisdientes-oral-diseases-4162
huggingface-lines-open-domain-oral-disease-qa
kaggle-bavithravairam-mouth-ulcer
kaggle-bavithravairam-oral-ulcer
kaggle-shlokmohanty-ulcer-classification
tongue-lesions-bmc-623patients
memosa-workbench-access-note
oii-ds-benchmark-wang-2023
osaka-prosthodontics-contact-draft-2026
roboflow-intra-oral-scanner-vgggs-r4xwu
roboflow-bitcamp-dental-j1vge
iter2-restorative-gap-note-secondary-caries
fdtooth-physionet-hku-2025
mendeley-4mxj6rpv48-facial-profile
schwarzmaier-143-ecc-anterior-deciduous-2024
aap-oral-health-image-gallery-pediatric
figshare-siop-29761292
zenodo-10580117-teledentistry-peru
jocpd-2026-011-pediatric-5view
sechenov-datamed-ai-platform-lead
datasetninja-dentalai-2495
github-tooth-segmentation44-yolov8
mendeley-ndb-ufes-bbmmm4wgr8-reference-only
aihub-korea-71509-dental-intraoral-clinical
iter3-gap-note-dentex-hhs9g-kaggle-caries-modality-unclear
iter3-gap-note-systematic-review-s41746-01818-5-datasets
roboflow-dental-plaque-sorbonne-teeth-detection-xdkru-3405
kaggle-santhoshsivang-calculus-dataset
ssrn-6281532-airc-labden-plaque-ortho
moharrami-ragadio-2024-gingivitis-666-kaggle-lead
outreach-di-gianfilippo-2025-recession-34-contact
outreach-lee-nycu-2025-keratinized-gingiva-576
outreach-kim-2025-plaque-quigley-hein-1394
outreach-nantakeeratipat-2024-plaque-600-thailand
outreach-vaughan-2025-multi-view-35-perio
outreach-wen-2024-gingivitis-826-sci-rep
iter3-perio-recession-gap-note-minimal
iter3-perio-rvg-v1301-blacklist-confirm
dfid-mltrmr-guizhou-131
integrated-fluorosis-grading-bspc-2024
mdpi-mih-oct-photonics-799-2025
frontiers-dde-mih-fluorosis-scoping-2025-1616109
coi-skin-dde-pediatric-2025-06326
frontiers-ai-witkop-ngs-1130175
mdpi-genes-16-822-2025
bmc-mixed-dentition-intraoral-2025-06866
cleft-mih-6432-coi-2025-06311
dryad-tcm-tongue-1c59zw48r
scidb-tongue-inquiry-8417299d
roboflow-ulcers-sw4n4
mpox-skin-lesion-v2-hfmd-subset
opmdcare-atlas-photos-2026
claseg-oral-lesions-2025-contact-note
multiclass-oral-mucosa-3246-1013-2025-contact-note
oralgpt-mucosa-xian-jiaotong-contact-note
roboflow-oralleukoplakia-418
alphadent-v1.2-release-check-negative-2026
aihub-kr-restorative-probe-2026
hf-mendeley-roboflow-api-probe-2026
ding-zhejiang-secondary-caries-outreach-2026
khurshid-bmc-2025-panoramic-prosthesis-out-of-scope-note
iter3-restorative-saturation-report
nyu-caredaway-sdf-school-ny-10620
peerj-20140-bengbu-occlusion-7200
osf-tf5r8-data-share-ortho
iter3-gap-note-ortho-pediatric
charite-berlin-5266-angle-lateral-2025
sunyatsen-zengcheng-pediatric-7671-2026
bengbu-7200-occlusion-classification-2025
osf-tgm5n-dental-ai-resources-list
sechenov-ai-205-adult-dental-photos-polygon
sechenov-ai-214-pediatric-dental-morbidity-moscow
bmc-hibogi-cimahi-indonesia-3221-2025
bmc-tbilisi-humanitarian-2864-georgia-2025
roboflow-aakash-npfax-dental-remote-areas-calculus-927
roboflow-abdul-aziz-caries-tartar-missing-6160
nanchang-university-kouqiang-3869-perio-screening-2025-lead
iter4-perio-peri-implantitis-photo-zero
iter4-perio-china-cn-sources-no-new
iter4-perio-mucogingival-graft-donor-gap
iter4-perio-gingiva-covid-gap
iter4-perio-saturation-flag
plos-one-0310420-mexican-fluorosis-mih-573
s41598-025-25960-y-vietnam-mih-1834-northern
dongthap-vietnam-mih-2025-40589688
abha-ksa-mih-520-kingkhalid-2026
mdpi-jcm-14-8959-enamel-caries-dl-2025
preprints-2025-tooth-wear-portuguese-community
iter4-gap-note-strasbourg-d4phenodent
iter4-gap-note-endemic-belts-no-public-photos
iter4-gap-note-betel-khat-chromogenic-no-open-photos
oralgpt-mmoral-dfull-dpartial-arxiv-2510-13911
kaggle-zaidpy-oral-cancer-dataset
kaggle-muhammadatef-oral-cancer-images-classification
arxiv-2511-21582-multimodal-oral-lesions-16class
iter4-gap-note-benign-inflammatory-no-dedicated-sets
roboflow-dental-qlzfr-dental-problems
iter4-surgical-zone-gap-structural
iter4-surgical-outreach-lead-polish-pediatric-cellulitis
iter4-surgical-outreach-lead-iadt-dentaltraumaguide
iter4-surgical-pmc-figure-harvest-plan
```

## FLAT ALIASES (for keyword exclusion)

```
10.5281/zenodo.14827784
zenodo:14827784
10.17632/9jnf2jvghy.2
mendeley:9jnf2jvghy
10.48550/arXiv.2507.22512
arxiv:2507.22512
mendeley:3253gj88rr
10.1186/s12903-024-05072-1
10.5281/zenodo.16582489
roboflow:image-segmentation-ltmbq/gingivitis-dataset-fdams
roboflow:digital-health-bg/gingivitis-t98xc
kaggle:salmansajid05/oral-diseases
github:PKNU-PR-ML-Lab/calculus
github:ZFTurbo/AlphaDent
10.1016/j.jds.2024.04.022
10.17632/6zsnhrds9t.1
mendeley:6zsnhrds9t
10.1007/s00784-022-04552-4
github:yunwu2024/dental_fluorosis
10.6084/m9.figshare.19641750
figshare:19641750
10.5281/zenodo.14622450
zenodo:14622450
10.1038/s41598-024-69125-9
github:wurenkai/HF-UNet-and-Autooral-dataset
10.5281/zenodo.10664056
10.5281/zenodo.14571990
zenodo:14571990
10.6084/m9.figshare.31341790
10.17632/mhjyrn35p4.2
mendeley:mhjyrn35p4
roboflow:kunchidsong-phosri/oral-lesion2
10.1038/s41597-024-04099-x
10.1038/s41598-021-81202-x
roboflow:havij/caries-tsrca-guzap
roboflow:dental-mate/crown-detection-mdfa5
roboflow:ai-in-dentistry/ai-in-dentistry-images-using-intraoral-cam-at-yang-dental
github:RoundFaceJ/OMNI
roboflow:orthodontic/orthodontic-noybl
github:thangngoc89/SegmentAnyTooth
arxiv:2407.11419
kaggle:alpha-dent/data
10.1038/s41415-025-9007-6
10.48550/arXiv.2511.04948
arxiv:2511.04948
10.1016/j.dib.2024.110926
10.1186/s12903-025-05803-y
10.3390/diagnostics16060862
10.1016/j.jds.2025.01.003
10.3390/dj14010060
10.1038/s41597-024-03836-6
10.17632/phtw6rmwzd.1
mendeley:phtw6rmwzd
roboflow:project-group13/dental-caries-detection-using-dl
roboflow:jesse-perring/caries-detection-pqmf0-io8pp
roboflow:bscs-8th-semester/tooth-cavity-detection
roboflow:yolo-v8-training/tooth-caries-3
roboflow:jb-okcsk/kaggle-dental-caries
kaggle:maazmakhdoom/dental-cavity-detection-dataset
github:liangyuandg/DLCariesScreen
10.1186/s12903-024-04460-x
10.1016/j.identj.2023.03.007
PMC12627268
10.3390/diagnostics15020231
roboflow:tooth-ytblb/dental-plaque
github:Priyanshu9898/Oral-Disease-Classification
roboflow:oral-disease-detection/teeth-disease-m1uob
10.2196/91239
10.3389/fphys.2023.1088703
10.1007/s10439-024-03615-9
10.6084/m9.figshare.30550889
figshare:30550889
10.5061/dryad.nvx0k6dxw
roboflow:tesisdientes/oral-diseases-5ctay-rqpxs
kaggle:bavithravairam/mouth-ulcer
kaggle:bavithravairam/oral-ulcer
kaggle:shlokmohanty/ulcer-classification
10.1186/s12880-024-01234-3
10.1016/j.compbiomed.2023.107620
roboflow:intraoral-scanner-vgggs/intra-oral-scanner-r4xwu
roboflow:bitcamp/dental-j1vge
10.13026/v9xk-dy61
10.17632/4mxj6rpv48.2
mendeley:4mxj6rpv48
10.3390/jcm13175215
10.6084/m9.figshare.29761292
figshare:29761292
10.5281/zenodo.10580117
zenodo:10580117
10.22514/jocpd.2026.011
github:YOLOv8-YOLOv11-Segmentation-Studio/tooth-segmentation44
10.17632/bbmmm4wgr8.4
mendeley:bbmmm4wgr8
roboflow:dentex-hhs9g/kaggle-caries
10.1038/s41746-025-01818-5
roboflow:dental-plaque-sorbonne/teeth-detection-xdkru
kaggle:santhoshsivang/calculus-dataset
10.1002/JPER.24-0173
PMC12123396
PMC12176457
10.1186/s12903-025-06350-2
PMC12220555
PMC11797812
PMC12274314
10.1038/s41598-024-70311-y
github:YOLOv8-YOLOv11-Segmentation-Studio/rvg-v1301
github:uxhao-o/MLTrMR
10.1016/j.bspc.2024.106829
10.3390/photonics12080799
10.3389/froh.2025.1616109
10.1007/s00784-025-06326-0
10.3389/fphys.2023.1130175
10.3390/genes16070822
10.1186/s12903-025-06866-7
10.1007/s00784-025-06311-7
10.5061/dryad.1c59zw48r
roboflow:ulcer-detection-sw4n4/ulcers
10.1016/j.bspc.2024.106742
github:mHealthBuet/Mpox-Skin-Lesion-Dataset-v2
10.1038/s41598-025-03268-1
arxiv:2510.13911
10.1186/s12903-025-07138-0
10.7717/peerj.20140
peerj:20140
10.1186/s12903-025-07550-6
PMC12861064
PMC12905764
PMC12478310
10.17605/OSF.IO/TGM5N
10.1186/s12903-025-07486-x
10.1186/s12903-025-06500-6
roboflow:aakash-npfax/dental-care-in-remote-areas
10.1002/rmv.70057
10.1371/journal.pone.0310420
10.1038/s41598-025-25960-y
10.3390/jcm14248959
PMC11725193
kaggle:zaidpy/oral-cancer-dataset
kaggle:muhammadatef/oral-cancer-images-for-classification
arxiv:2511.21582
PMC11909381
roboflow:dental-qlzfr/dental-problems
PMC10049628
```