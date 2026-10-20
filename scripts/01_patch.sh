#!/usr/bin/env bash
# Tissue segmentation and tiling. Discovery cohort: 256 px at pyramid level 1 (20x, ~0.5 um/px),
# tissue mask from the level-6 thumbnail (cancerscout/tile_cancerscout.py). CPTAC: CLAM, 256 px at
# level 0 (0.25 um/px), preset tcga.csv (CLAM's default segmentation: sthresh 8, mthresh 7, close 4,
# a_t 16); step 07 turns this lattice into 512 px (20x) crops.
# COHORTS selects the cohorts (default: both).
source "$(dirname "$0")/common.sh"
cd "$CLAM"
for cohort in ${COHORTS:-cancerscout cptac}; do
  if [ "$cohort" = cancerscout ]; then
    need_cancerscout
    $PY "$REPO/cancerscout/tile_cancerscout.py" --source "$CSWSI_SLIDES_CS" --save_dir "$CSWSI_WORK/patches/cancerscout"
  else
    need_cptac
    $PY create_patches_fp.py --source "$CSWSI_SLIDES_CPTAC" --save_dir "$CSWSI_WORK/patches/cptac" \
        --patch_size 256 --step_size 256 --patch_level 0 --preset tcga.csv --seg --patch --stitch
  fi
done
