#!/usr/bin/env bash
# Fig. 5E,F; Extended Data Figs 9A-D and 10A-D; written to $CSWSI_WORK/figures (PDF and PNG).
# Run 11_heatmaps.sh for Fig. 5D and Supplementary Figs 8 and 9.
# PANELS=cptac draws only the panels that need no discovery-cohort data (Fig. 5F, ED Fig. 10B-D).
source "$(dirname "$0")/common.sh"
cd "$REPO/figures"
if [ "${PANELS:-all}" = cptac ]; then
  list="fig5F_cptac_transfer_20x.py ed10B_tumor_calls_20x.py cptac_bacc_thresholds_20x.py"
else
  list="fig5E_ed9BC_task_overview.py fig5F_cptac_transfer_20x.py ed9A_roc.py ed9D_subgroup_composition.py
        ed10A_domain_gap_20x.py ed10B_tumor_calls_20x.py cptac_bacc_thresholds_20x.py"
fi
for f in $list; do $PY "$f"; done
