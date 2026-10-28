#!/usr/bin/env bash
# Trained CLAM models, their decision thresholds and StainNet from Zenodo into models/ (checksums checked).
# ENCODERS selects a subset; MODEL_ARCHIVES=<dir> takes archives already on disk instead of downloading.
source "$(dirname "$0")/common.sh"
$PY "$REPO/scripts/download_models.py" ${MODEL_ARCHIVES:+--from-dir "$MODEL_ARCHIVES"}
