#!/usr/bin/env bash
# Per-fold and consensus metrics, raw vs StainNet tests on CPTAC, Supplementary Table 6.
# Writes to $CSWSI_WORK/tables. Without the held-out predictions (CPTAC-only run) the CPTAC tables
# are still written and Supplementary Table 6 is skipped.
source "$(dirname "$0")/common.sh"
cd "$REPO/analysis"
$PY compute_numbers.py
$PY normalization_tests.py
$PY bacc_tests_20x.py
$PY normalization_effects_20x.py
$PY threshold_migration_20x.py
$PY agreement_20x.py
if [ -n "$(ls "$CSWSI_EVAL_DIR"/EVAL_cancerscout_colon_he_*/fold_0.csv 2>/dev/null)" ]; then
  $PY build_supp_table6.py
  $PY build_supp_table6_portrait.py
fi
