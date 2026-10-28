# sourced by every step: paths, encoders, endpoints
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/configs/paths.env" ] && { set -a; source "$REPO/configs/paths.env"; set +a; }
# defaults mirror cswsi_config.py; override any of them in configs/paths.env
: "${CSWSI_WORK:=$REPO/work}" "${CSWSI_MODELS:=$REPO/models}"
: "${CSWSI_SLIDES_CS:=$REPO/data/cancerscout/slides}" "${CSWSI_LABELS:=$REPO/data/cancerscout/labels}"
: "${CSWSI_SLIDES_CPTAC:=$REPO/data/cptac/slides}" "${CSWSI_EVAL_DIR:=$CSWSI_WORK/eval_results}"
export CSWSI_WORK CSWSI_MODELS CSWSI_SLIDES_CS CSWSI_LABELS CSWSI_SLIDES_CPTAC CSWSI_EVAL_DIR
PY="${PYTHON:-python}"
GPU="${GPU:-0}"
CLAM="$REPO/clam"
SPLITS="$CSWSI_WORK/splits"
MODELS_CLAM="$CSWSI_MODELS/clam"
# encoder short name -> CLAM model name, embedding size
declare -A MODEL=( [resnet]=resnet50_trunc [uni]=uni_v1 [conch]=conch_v1 )
declare -A DIM=( [resnet]=1024 [uni]=1024 [conch]=512 )
ENCODERS="${ENCODERS:-resnet uni conch}"
# endpoints.tsv rows without the header
endpoints() { tail -n +2 "$REPO/configs/endpoints.tsv"; }
# fail early with a pointer to the README of the missing input
need_dir() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ] || { echo "missing or empty: $1 (see $2)" >&2; exit 1; }; }
need_cancerscout() { need_dir "$CSWSI_SLIDES_CS" data/cancerscout/README.md; need_dir "$CSWSI_LABELS" data/cancerscout/README.md; }
need_cptac() { need_dir "$CSWSI_SLIDES_CPTAC" data/cptac/README.md; }
need_models() { need_dir "$MODELS_CLAM" "models/README.md; get them with: bash scripts/download_models.sh"; }
need_encoders() {
  for enc in $ENCODERS; do
    case $enc in uni) : "${UNI_CKPT_PATH:?set UNI_CKPT_PATH (see README, Encoder weights)}";;
                 conch) : "${CONCH_CKPT_PATH:?set CONCH_CKPT_PATH (see README, Encoder weights)}";; esac
  done
}
