#!/usr/bin/env bash
# The frozen CancerScout models on CPTAC, no retraining and no threshold recalibration. The
# per-model thresholds (Youden J on each fold's internal test split) are written first; they come
# from the training run stored with each model, so this step needs no CancerScout data.
source "$(dirname "$0")/common.sh"
need_models
cd "$CLAM"
$PY "$REPO/analysis/validation_thresholds.py"
declare -A SUFFIX=( [scale20x]=scale20x_full [scale20x_stainnet]=scale20xsn_full )
endpoints | while IFS=$'\t' read -r key dataset multi intl cpt; do
  [ "$cpt" = 1 ] || continue
  for enc in $ENCODERS; do
    for cond in scale20x scale20x_stainnet; do
      CUDA_VISIBLE_DEVICES=$GPU $PY eval_cancerscout.py --task generic --split all --k 10 \
          --model_type clam_mb --model_size small --drop_out 0.25 --results_dir "$MODELS_CLAM" --eval_root "$CSWSI_EVAL_DIR" \
          --models_exp_code "cancerscout_colon_he_${dataset}_v3_${enc}_s1" \
          --save_exp_code "cancerscout_colon-cptac_${key}_v3_${enc}_${SUFFIX[$cond]}" \
          --data_root_dir "$CSWSI_WORK/features/cptac/${enc}_${cond}" \
          --wsi_list_path "${CSWSI_CPTAC_LABELS:-$REPO/configs/cptac_labels}/${key}.csv" --embed_dim "${DIM[$enc]}"
    done
  done
done
