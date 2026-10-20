# Methods as implemented

This describes what the code does, step by step, with the settings used for the paper. The script
that implements each part is given in brackets.

## Cohorts and endpoints

**Discovery cohort.** H&E whole-slide images (WSIs) of the CancerScout colon cohort, 2,496 slides
used in the benchmarks. Slide-level labels come from patient-level clinical and molecular data
(`configs/endpoints.tsv`, 16 endpoints): tumor detection (tumor-containing vs tumor-free sections
of cohort patients), sex, grade (three classes: G2, G3 and MSI tumors, which carry no grade),
lymph-node and distant metastasis, CPTAC subtype (CIN, MSI, mesenchymal), MSI status from
immunohistochemistry, from genetics and their consensus, BRAF, KRAS/NRAS, TP53 and APC mutation
status, the six proteomic subgroups (C1-C6), the three CIN subgroups C2/C3/C5 among CIN cases, and
CIN versus mesenchymal class.

**Patient-level split.** Each endpoint's labels are split at the patient level into a training set
(70% of patients, `train.csv`) and a held-out evaluation set (30%, `eval.csv`); no patient has slides
on both sides. The held-out set is used only for evaluation. Every in-domain number in the paper
comes from it.

**CPTAC-COAD.** All 372 H&E tissue slide images of the TCIA collection (178 patients, all
tumor-positive). MSI status is the sequencing-based call (`MSmutectFishercall`; 223 slides from 106
patients) and the CPTAC subtype the UMS subtype (190 slides from 87 patients), both from Vasaikar et
al. (Cell 2019), assigned to every slide of a patient (`configs/cptac_labels/`).

## Tiling and features

* **Discovery-cohort tiling** (`cancerscout/tile_cancerscout.py`): non-overlapping 256 px tiles at
  pyramid level 1 (about 0.5 um/px, 20x; 256 px = about 128 um). The tissue mask is computed on the
  level-6 thumbnail: the scanner's flat red fill is set to white; background is bright (mean RGB
  > 210) and texture-free (median-blurred absolute Laplacian <= 5) after a binary opening; the mask
  is resized to the tile grid, inverted, and tissue islands and holes smaller than 20 tiles are
  removed. Every tissue tile is kept.
* **CPTAC tiling** with CLAM (`clam/create_patches_fp.py`, preset `tcga.csv`, CLAM's default
  segmentation): 256 px tiles at level 0.
* **Encoders**, frozen, no fine-tuning (`scripts/02_features.sh`): ResNet50 pretrained on ImageNet
  (truncated after the third block, 1,024-d), UNI (1,024-d) and CONCH (vision encoder, 512-d). Tiles
  are resized to 224 px.
* **CPTAC at the matched resolution** (`cptac/reextract_20x.py`). CPTAC slides are scanned at 40x
  (0.25 um/px) and tiled at level 0 with 256 px tiles. Every second row and column of that lattice is
  kept, and a 512 px level-0 crop is read at each kept coordinate: the same 128 um field of view and
  the same non-overlapping layout as a discovery-cohort tile. The crop is resized to 224 px by the
  encoder's transform. Two conditions: raw pixels, and StainNet-normalized pixels.

## Stain normalization (CPTAC to discovery cohort)

* **Teacher**: Macenko normalization with fixed statistics (`stain_normalization/macenko.py`). The
  target stain matrix and maximum concentrations are fitted once on three discovery-cohort reference
  tiles; the source stain matrix and maximum once on ten CPTAC tiles pooled together
  (256 px at level 0, positions in `configs/stainnet_source_pool.csv`). The same transform is applied to every tile, so the
  teacher is a single color map (no per-tile refit).
* **StainNet** (Kang et al. 2021; `stain_normalization/stainnet.py`): three 1x1 convolution layers
  with 32 channels, a per-pixel color map with no spatial context. Trained to reproduce the teacher on
  2,000 CPTAC tiles (40 random 256 px tiles from each of 50 slides, `configs/stainnet_training_slides.csv`):
  L1 loss on [-1, 1]-scaled images, Adam (learning rate 2e-4, cosine annealing), batch 32, 25 epochs
  (`build_stainnet_dataset.py`, `train_stainnet.py`). The 50 training slides are also among the
  evaluated CPTAC slides; StainNet learns only a color map, not labels.
* **Staining difference** (Extended Data Fig. 10A): mean per-channel Wasserstein distance between
  the RGB intensity distributions of tissue pixels, one random tissue tile per discovery-cohort slide
  and four per CPTAC slide, 1,500 tissue pixels per tile, CPTAC raw or StainNet vs discovery cohort.

## Models

* **CLAM-MB** (Lu et al. 2021), small configuration, dropout 0.25, Adam (learning rate 2e-4, weight
  decay 1e-5), cross-entropy bag loss with a smooth SVM instance loss (B = 8, bag weight 0.7),
  class-balanced sampling, early stopping on the validation loss (patience 20 epochs from epoch 50,
  at most 200 epochs) (`scripts/04_train.sh`, `clam/main.py`).
* **Ten models per endpoint and encoder** (`scripts/03_splits.sh`): ten random draws inside the
  training set, stratified by class: 15% of slides for validation (early stopping), 15% for an
  internal test split, and a random 70% of the remaining slides for training (CLAM's `label_frac`
  0.7; seed 1). The draws are random at the slide level, so sections of
  one patient can fall in different parts of a draw; this affects only model selection and
  thresholds, never the reported numbers, which come from the patient-level held-out set.
* **Decision thresholds** (`analysis/validation_thresholds.py`): for each binary model, the
  threshold maximizing Youden's J on that model's internal test split. Multiclass models are called
  by argmax. The thresholds are carried unchanged to CPTAC (no recalibration).

## Evaluation

* **Per-model metrics**: AUC (binary: ROC AUC of the positive-class probability; multiclass:
  macro-averaged one-vs-rest) and balanced accuracy at the model's threshold
  (`analysis/compute_numbers.py`). Several rescans of one glass slide that point to the same
  feature file are counted once.
* **Ten-model combination**: AUC of the probability averaged over the ten models (soft vote), and
  balanced accuracy of the majority vote of the ten per-model calls, ties broken by the class with
  the highest averaged probability (`analysis/consensus_vote.py`). The tables also report the
  prevalence-weighted one-vs-rest AUC (`auc_consensus_weighted`).
* **CPTAC transfer**: the same ten models of each endpoint, frozen, on every CPTAC slide with a label,
  raw and StainNet (`scripts/08_eval_cptac.sh`). Tumor detection has only positive slides, so it is
  reported as the share of slides called tumor.
* **Raw vs StainNet** (paired, same slides and models):
  * AUC of the averaged prediction: case-level paired bootstrap (2,000 resamples of CPTAC patients,
    all slides of a patient kept together), 95% CI and two-sided p; Wilcoxon signed-rank over the ten
    per-model AUCs (`analysis/normalization_tests.py`);
  * balanced accuracy of the majority vote: patient bootstrap (2,000 resamples); tumor calls: exact
    McNemar test (`analysis/bacc_tests_20x.py`);
  * one Holm correction over all raw-vs-StainNet comparisons (`analysis/normalization_effects_20x.py`);
  * agreement of the ten models with their majority vote, per slide (`analysis/agreement_20x.py`);
  * how far each model's transferred threshold is from the best threshold on CPTAC (an oracle shown
    for interpretation only, `analysis/threshold_migration_20x.py`).
* **Attention maps** (`scripts/11_heatmaps.sh`): tile attention of the CONCH models drawn over the
  H&E thumbnail. Held-out examples use, per slide and task, the model listed in
  `configs/example_slides.yaml`; CPTAC examples use one pre-specified model (fold 8) for every slide
  and task, with the ten-model majority vote printed above each map.
