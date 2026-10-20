# Reproducing the paper's WSI results

Every panel and table below is written to `work/figures/` or `work/tables/` (or `$CSWSI_FIGURES`,
`$CSWSI_TABLES`). "CPTAC-only" marks what `scripts/run_cptac.sh` reproduces from public data and
the trained models alone, without the discovery-cohort release.

## Figures

| Paper panel | Script | Output file | Inputs | CPTAC-only |
|---|---|---|---|---|
| Fig. 5D | `figures/fig5D_suppfig8_attention_grids.py` (via `scripts/11_heatmaps.sh`) | `Fig5D_attention_C2` | CONCH attention maps of the C2 example; H&E slide; labels | no |
| Fig. 5E | `figures/fig5E_ed9BC_task_overview.py` | `Fig5E_heldout_AUC` | `in_domain_metrics.csv`, `in_domain_perfold.csv` | no |
| Fig. 5F | `figures/fig5F_cptac_transfer_20x.py` | `Fig5F_CPTAC_20x_StainNet_AUC` | `cptac_metrics.csv`, `cptac_perfold.csv` | yes |
| Extended Data Fig. 9A | `figures/ed9A_roc.py` | `ExtDataFig9A_heldout_ROC` | held-out predictions (`work/eval_results/EVAL_cancerscout_colon_he_*`) | no |
| Extended Data Fig. 9B,C | `figures/fig5E_ed9BC_task_overview.py` | `ExtDataFig9B_heldout_AUC`, `ExtDataFig9C_heldout_balanced_accuracy` | as Fig. 5E | no |
| Extended Data Fig. 9D | `figures/ed9D_subgroup_composition.py` | `ExtDataFig9D_subgroup_composition` | held-out predictions of the six-class models; `labels/annotations/cptac_subtype_per_slide.csv` | no |
| Extended Data Fig. 10A | `figures/ed10A_domain_gap_20x.py` | `ExtDataFig10A_domain_gap_20x` | slides and tile coordinates of both cohorts; StainNet | no |
| Extended Data Fig. 10B | `figures/ed10B_tumor_calls_20x.py` | `ExtDataFig10B_CPTAC_tumor_calls_20x` | CPTAC tumor-detection predictions; thresholds | yes |
| Extended Data Fig. 10C,D | `figures/cptac_bacc_thresholds_20x.py` | `ExtDataFig10CD_CPTAC_balanced_accuracy_20x` | `cptac_perfold.csv`, `cptac_metrics.csv` | yes |
| Supplementary Fig. 8 | `figures/fig5D_suppfig8_attention_grids.py` (via `scripts/11_heatmaps.sh`) | `SuppFig8_attention_by_subgroup` | CONCH attention maps of six held-out examples (C1-C6) | no |
| Supplementary Fig. 9 | `figures/suppfig9_cptac_attention_20x.py` (via `scripts/11_heatmaps.sh`) | `SuppFig9_CPTAC_attention_20x` | CONCH attention maps of five CPTAC slides (fold 8); CPTAC predictions | yes |

`cptac_bacc_thresholds_20x.py` also writes `CPTAC_20x_balanced_accuracy_and_threshold_shift`, an
extended version with the threshold-shift panels (analysis only, not in the paper).

## Tables

| Output (`work/tables/`) | Script | Content | CPTAC-only |
|---|---|---|---|
| `in_domain_metrics.csv`, `in_domain_perfold.csv` | `analysis/compute_numbers.py` | held-out AUC and balanced accuracy, per model and ten-model combination | no |
| `cptac_metrics.csv`, `cptac_perfold.csv` | `analysis/compute_numbers.py` | the same on CPTAC, raw and StainNet | yes |
| `normalization_tests_20x.csv` | `analysis/normalization_tests.py` | raw vs StainNet: AUC of the averaged prediction, tumor calls | yes |
| `bacc_tests_20x.csv` | `analysis/bacc_tests_20x.py` | raw vs StainNet: balanced accuracy of the majority vote, tumor calls | yes |
| `normalization_effects_20x.csv` | `analysis/normalization_effects_20x.py` | all raw-vs-StainNet comparisons, one Holm correction | yes |
| `agreement_20x.csv` | `analysis/agreement_20x.py` | agreement of the ten models, raw vs StainNet | yes |
| `cptac_threshold_migration_20x.csv` | `analysis/threshold_migration_20x.py` | transferred vs best CPTAC threshold per model | yes |
| `SupplementaryTable6.{csv,xlsx}`, `SupplementaryTable6_portrait.docx` | `analysis/build_supp_table6.py`, `build_supp_table6_portrait.py` | Supplementary Table 6 | no |
| `ed10A_wasserstein_20x.csv` | `figures/ed10A_domain_gap_20x.py` | Wasserstein distances behind Extended Data Fig. 10A | no |

## Numbers quoted in the text

| Text | Source |
|---|---|
| Held-out AUC and balanced accuracy per task and encoder | `in_domain_metrics.csv` (`auc_consensus`, `bacc_consensus`), Supplementary Table 6 |
| CPTAC AUC for MSI status and CPTAC subtype (raw, StainNet) | `cptac_metrics.csv` |
| Share of CPTAC slides called tumor; change with StainNet | `bacc_tests_20x.csv`, `normalization_effects_20x.csv` |
| No significant raw-vs-StainNet AUC difference | `normalization_effects_20x.csv` (Holm-adjusted p) |
| Ten models agree more often after normalization | `agreement_20x.csv` |
| Wasserstein distance 46.8 raw vs 10.3 StainNet | `ed10A_wasserstein_20x.csv` |

## Checks

Verified against the study's stored outputs before release:

* `configs/cptac_manifest.tsv` lists the 372 CPTAC files with size and MD5; it matches the TCIA
  package listing (names and sizes), and step 00 verifies every download against it.
* CPTAC route, end to end on two slides from the raw SVS files: CLAM tiling (preset `tcga.csv`)
  gives the stored tile coordinates; the 20x features of all three encoders, raw and StainNet, are
  bit-identical to the stored ones; the predictions of all ten models of every CPTAC endpoint are
  identical.
* The ten CPTAC tiles that fix the StainNet teacher's source statistics, read from the slides at
  the positions in `configs/stainnet_source_pool.csv`, are pixel-identical to the tiles used.
* From the trained models alone, the per-model decision thresholds (step 08) equal the ones used in
  the paper (36 of 36 binary model sets).
* From the stored predictions, steps 09-11 reproduce every table and every panel listed above
  exactly (tables byte for byte, figures pixel for pixel), including Supplementary Table 6.
* Discovery-cohort tiling: `cancerscout/tile_cancerscout.py` gives the same tiles as the original
  tiling code run today; against the tiles of the 2023 run, 0-0.5% of tiles per slide differ
  (7 slides checked), which we attribute to library versions at the time. GPU steps are otherwise
  deterministic up to floating-point differences between GPU models and library versions.
