#!/usr/bin/env bash
# All ten models of every endpoint and encoder on the patient-level held-out set (eval.csv).
source "$(dirname "$0")/common.sh"
need_cancerscout; need_models
cd "$CLAM"
endpoints | while IFS=$'\t' read -r key dataset multi intl cpt; do
  for enc in $ENCODERS; do
    extra=""; [ "$intl" = 1 ] && extra="--int_label_keys"
    CUDA_VISIBLE_DEVICES=$GPU $PY eval_cancerscout.py --task generic --split all --k 10 \
        --model_type clam_mb --model_size small --drop_out 0.25 --results_dir "$MODELS_CLAM" --eval_root "$CSWSI_EVAL_DIR" \
        --models_exp_code "cancerscout_colon_he_${dataset}_v3_${enc}_s1" \
        --save_exp_code "cancerscout_colon_he_${key}_v3_${enc}_s1" \
        --data_root_dir "$CSWSI_WORK/features/cancerscout/$enc" \
        --wsi_list_path "$CSWSI_LABELS/$dataset/eval.csv" --embed_dim "${DIM[$enc]}" $extra
  done
done
# per-fold decision thresholds (Youden J on each fold's internal test split of the training set)
$PY "$REPO/analysis/validation_thresholds.py"
