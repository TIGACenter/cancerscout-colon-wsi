#!/usr/bin/env bash
# The external CPTAC-COAD evaluation end to end, with no discovery-cohort data: download and verify
# the slides, tile them, embed them at 20x (raw and StainNet), apply the trained models (not part of
# this repository; see models/README.md) and compute the CPTAC tables and panels: Fig. 5F, Extended
# Data Fig. 10B-D and Supplementary Fig. 9, plus the raw-vs-StainNet tests.
# Each step can also be run on its own; GPU selects the device (default 0).
set -euo pipefail
D="$(dirname "$0")"
bash "$D/00_download_cptac.sh"
COHORTS=cptac bash "$D/01_patch.sh"
bash "$D/07_cptac_features_20x.sh"
bash "$D/08_eval_cptac.sh"
bash "$D/09_metrics.sh"
PANELS=cptac bash "$D/10_figures.sh"
PANELS=cptac bash "$D/11_heatmaps.sh"
