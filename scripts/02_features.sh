#!/usr/bin/env bash
# Frozen tile embeddings (224 px input) for the three encoders. UNI and CONCH weights are gated on
# Hugging Face: set UNI_CKPT_PATH and CONCH_CKPT_PATH.
source "$(dirname "$0")/common.sh"
need_cancerscout; need_encoders
cd "$CLAM"
for enc in $ENCODERS; do
  CUDA_VISIBLE_DEVICES=$GPU $PY extract_features_fp.py --data_h5_dir "$CSWSI_WORK/patches/cancerscout" \
      --data_slide_dir "$CSWSI_SLIDES_CS" --slide_ext .tiff \
      --csv_path "$CSWSI_WORK/patches/cancerscout/process_list_autogen.csv" \
      --feat_dir "$CSWSI_WORK/features/cancerscout/$enc" --model_name "${MODEL[$enc]}" \
      --batch_size 256 --target_patch_size 224
done
