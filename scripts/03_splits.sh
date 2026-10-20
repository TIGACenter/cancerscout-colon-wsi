#!/usr/bin/env bash
# Ten random train/validation/test draws inside each endpoint's training set (patient-level
# held-out set untouched): 15% validation, 15% test, 70% of the rest for training, seed 1.
source "$(dirname "$0")/common.sh"
need_cancerscout
cd "$CLAM"
endpoints | while IFS=$'\t' read -r key dataset multi intl cpt; do
  $PY create_splits_cancerscout.py --task generic --wsi_list_path "$CSWSI_LABELS/$dataset/train.csv" \
      --splits_root "$SPLITS" --seed 1 --label_frac 0.7 --k 10 --val_frac 0.15 --test_frac 0.15 \
      --save_dir_name "cancerscout_colon_he_${dataset}_v3"
done
