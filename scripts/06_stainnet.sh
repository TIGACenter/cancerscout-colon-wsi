#!/usr/bin/env bash
# StainNet: a per-pixel network distilled from a fixed-statistics Macenko teacher. Needs the CPTAC
# slides and tiling (steps 00, 01), the ten CPTAC source-pool tiles (read from the
# slides at the positions in configs/stainnet_source_pool.csv) and the three discovery-cohort target tiles of the data
# release ($CSWSI_STAIN_REFERENCE, default data/cancerscout/stain_reference/).
# 2,000 CPTAC tiles (40 from each of 50 slides) paired with a fixed-statistics Macenko target
# toward three CancerScout reference tiles; L1 loss, Adam 2e-4, batch 32, 25 epochs.
source "$(dirname "$0")/common.sh"
cd "$REPO/stain_normalization"
$PY build_stainnet_dataset.py
CUDA_VISIBLE_DEVICES=$GPU $PY train_stainnet.py
mkdir -p "$CSWSI_MODELS/stainnet"
cp "$CSWSI_WORK/stainnet/checkpoints/stainnet_final.pt" "$CSWSI_MODELS/stainnet/stainnet_final.pt"
