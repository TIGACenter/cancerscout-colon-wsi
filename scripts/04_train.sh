#!/usr/bin/env bash
# CLAM-MB, small, dropout 0.25, Adam lr 2e-4, weight decay 1e-5, cross-entropy bag loss with SVM
# instance loss (B = 8, bag weight 0.7), class-balanced sampling, early stopping (patience 20 from
# epoch 50, max 200 epochs), ten models per endpoint and encoder.
source "$(dirname "$0")/common.sh"
need_cancerscout
cd "$CLAM"
endpoints | while IFS=$'\t' read -r key dataset multi intl cpt; do
  for enc in $ENCODERS; do
    extra=""; [ "$multi" = 1 ] && extra="--subtyping"
    CUDA_VISIBLE_DEVICES=$GPU $PY main.py --early_stopping --lr 2e-4 --reg 1e-5 --drop_out 0.25 \
        --weighted_sample --bag_loss ce --inst_loss svm --B 8 --bag_weight 0.7 --max_epochs 200 --k 10 --seed 1 \
        --results_dir "$MODELS_CLAM" --model_type clam_mb --model_size small --log_data --task generic --embed_dim "${DIM[$enc]}" \
        --exp_code "cancerscout_colon_he_${dataset}_v3_${enc}" \
        --data_root_dir "$CSWSI_WORK/features/cancerscout/$enc" \
        --wsi_list_path "$CSWSI_LABELS/$dataset/train.csv" \
        --split_dir "$SPLITS/cancerscout_colon_he_${dataset}_v3_70" $extra
  done
done
