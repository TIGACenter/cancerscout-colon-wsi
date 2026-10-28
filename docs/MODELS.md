# Model card: CancerScout colon WSI models

Trained models of *A proteogenomic resource of 1,000 colorectal cancers resolves functional tumor
states* (Qoku\*, Ullrich\*, Miranda\* et al.), archived on Zenodo (DOI in `configs/zenodo.env`) and
downloaded with `scripts/download_models.sh` (`models/README.md`).

## What the models are

* **CLAM-MB** attention-based multiple-instance classifiers (Lu et al. 2021), small configuration:
  each takes the bag of tile embeddings of one H&E whole-slide image and returns slide-level class
  probabilities and per-tile attention.
* **Inputs**: tile embeddings of 256 px tissue tiles at 20x (about 0.5 um/px, 128 um field of view),
  resized to 224 px, from a frozen encoder: ResNet50 (ImageNet, truncated after the third block,
  1,024-d), UNI (1,024-d) or CONCH (vision encoder, 512-d). The pipeline in this repository produces
  them (`scripts/01_patch.sh`, `scripts/02_features.sh`; CPTAC: `scripts/07_cptac_features_20x.sh`).
* **Ten models per endpoint and encoder** (folds 0-9), trained on ten random draws inside the same
  training set; the paper reports each model and their combination (averaged probability, majority
  vote of the ten calls).
* **StainNet** (Kang et al. 2021): a per-pixel color map (three 1x1 convolutions) from CPTAC-COAD
  colors to discovery-cohort colors, used only for the CPTAC evaluation.

Training settings, splits and evaluation are in `docs/METHODS.md`.

## Endpoints and classes

Output index order is given by `label_dict.json` in each model folder and is the same for the three
encoders. For binary models, the stored threshold applies to the probability of class index 1.

| `<dataset>` | Endpoint | Classes by output index | Class 1 (thresholded) | Evaluated on CPTAC |
|---|---|---|---|---|
| `tumor_vs_normal` | tumor detection | 0 tumor, 1 tumor-free | tumor-free | yes |
| `sex` | sex | 0 male, 1 female | female | |
| `grading` | grade | 0 G3, 1 G2, 2 MSI (no grade) | argmax | |
| `ln_metastasis` | lymph-node metastasis | 0 yes, 1 no | no | |
| `distant_metastasis` | distant metastasis | 0 no, 1 yes | yes | |
| `cptac` | CPTAC subtype | 0 MSI, 1 CIN, 2 mesenchymal | argmax | yes |
| `msi_status` | MSI status, immunohistochemistry | 0 MSS, 1 MSI | MSI | yes |
| `msi_status_genetic` | MSI status, genetics | 0 MSS, 1 MSI | MSI | yes |
| `msi_status_consensus` | MSI status, consensus | 0 MSS, 1 MSI | MSI | yes |
| `braf` | BRAF | 0 mutated, 1 wild type | wild type | |
| `k_n_ras` | KRAS/NRAS | 0 mutated, 1 wild type | wild type | |
| `tp53` | TP53 | 0 mutated, 1 wild type | wild type | |
| `apc` | APC | 0 mutated, 1 wild type | wild type | |
| `proteomic_classification` | proteomic subgroup | 0 C1, 1 C2, 2 C4, 3 C3, 4 C5, 5 C6 | argmax | |
| `proteomic_groups_c2c3c5_only_cin_cases` | CIN subgroup, CIN cases | 0 C2, 1 C3, 2 C5 | argmax | |
| `cin_vs_mcml` | CIN vs mesenchymal | 0 CIN, 1 mesenchymal | mesenchymal | |

Note the proteomic index order: C4 is index 2 and C3 index 3.

## Training data

H&E whole-slide images of the CancerScout colon discovery cohort (`docs/METHODS.md`, Cohorts and
endpoints): for each endpoint, the training set of a patient-level 70/30 split; the 30% held-out set
was never used for training, model selection or thresholds. The slides and labels are released in
the BioImage Archive on publication (`data/cancerscout/README.md`). StainNet was trained on 2,000
CPTAC-COAD tiles (50 slides) against a fixed-statistics Macenko target fitted on three
discovery-cohort tiles; it learns only a color map.

## Verification of this release

The archived checkpoints are the ones behind the paper's numbers. Before release, each of the 48
endpoint-encoder sets was identified by rerunning its ten models on held-out slides and comparing
with the stored predictions of the paper (all matched), and every threshold in
`thresholds_<encoder>.csv` was refitted from the training runs and compared with the stored one
(all 360 matched). Some endpoints come from the training runs of November 2024 to March 2025
(tumor detection, sex, lymph-node metastasis, MSI status with ResNet50 and UNI, MSI genetics and
consensus), the others from May 2026; both are the versions the paper reports.

## Intended use and limits

* Research use: reproducing and extending the paper's analyses. Not a medical device; not validated
  for diagnosis or treatment decisions.
* Trained on one cohort, with one scanner and laboratory. On CPTAC-COAD the ranking of slides
  (AUC) carried over better than the decisions at the stored thresholds; thresholds may need local
  recalibration on a new cohort (paper, Extended Data Fig. 10).
* The UNI and CONCH models need the corresponding encoders under MahmoodLab's terms.

## License and citation

Weights: CC BY 4.0 (the UNI and CONCH encoders they need stay under MahmoodLab's terms). Please cite the paper and the Zenodo record, doi:10.5281/zenodo.23669359.
