#!/usr/bin/env bash
# CPTAC at the training resolution: 512 px level-0 crops (128 um, as CancerScout's 256 px at 20x),
# raw and StainNet-normalized, for the three encoders.
source "$(dirname "$0")/common.sh"
need_cptac; need_encoders
ls "$CSWSI_SLIDES_CPTAC" | sed -n 's/\.svs$//p' > "$CSWSI_WORK/cptac_slides.txt"
for enc in $ENCODERS; do
  for cond in scale20x scale20x_stainnet; do
    CUDA_VISIBLE_DEVICES=$GPU $PY "$REPO/cptac/reextract_20x.py" --slide_list "$CSWSI_WORK/cptac_slides.txt" \
        --model_name "${MODEL[$enc]}" --condition "$cond" --skip_existing
  done
done
