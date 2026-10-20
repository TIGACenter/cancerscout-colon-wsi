#!/usr/bin/env bash
# The full study: discovery-cohort training and held-out evaluation, then the CPTAC transfer.
# Needs the discovery-cohort release in data/cancerscout/ (see data/cancerscout/README.md).
set -euo pipefail
D="$(dirname "$0")"
bash "$D/00_download_cptac.sh"
for s in 01_patch 02_features 03_splits 04_train 05_eval_heldout 06_stainnet 07_cptac_features_20x \
         08_eval_cptac 09_metrics 10_figures 11_heatmaps; do
  bash "$D/$s.sh"
done
