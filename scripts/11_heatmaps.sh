#!/usr/bin/env bash
# CLAM attention maps of the CONCH models: held-out discovery examples (Fig. 5D, Supplementary Fig. 8)
# and CPTAC at 20x after StainNet (Supplementary Fig. 9, one pre-specified model: fold 8). Example
# slides, the fold of each published held-out map and crops: configs/example_slides.yaml.
# PANELS=cptac renders only Supplementary Fig. 9 (no discovery-cohort data needed).
source "$(dirname "$0")/common.sh"
need_models
cd "$CLAM"
PANELS="${PANELS:-all}"
PL="$CSWSI_WORK/heatmaps/process_lists"; mkdir -p "$PL"
$PY - "$REPO" "$PL" "$CSWSI_LABELS" "$PANELS" <<'PYEOF'
import sys, yaml, pandas as pd
repo, pl, labels, panels = sys.argv[1:]
ex = yaml.safe_load(open(f'{repo}/configs/example_slides.yaml'))
if panels != 'cptac':
    cs = {'proteomic_classification': 'proteomic_classification', 'cptac': 'cptac',
          'tumor_vs_normal': 'tumor_vs_normal', 'msi_status_consensus': 'msi_status_consensus'}
    folds = open(f'{pl}/cs_folds.tsv', 'w')
    for code, slide in ex['cancerscout_subgroup_examples'].items():
        for task, ds in cs.items():
            folds.write(f"{task}\t{slide}\t{ex['cancerscout_example_folds'][code][task]}\n")
            lab = pd.concat([pd.read_csv(f'{labels}/{ds}/{s}.csv') for s in ('train', 'eval')])
            lab[lab.slide_id == slide].head(1).to_csv(f'{pl}/cs__{task}__{slide}.csv', index=False)
for slide in ex['cptac_examples']:
    for task in ('tumor_vs_normal', 'cptac', 'msi_status'):
        lab = pd.read_csv(f"{__import__('os').environ.get('CSWSI_CPTAC_LABELS', repo + '/configs/cptac_labels')}/{task}.csv")
        lab[lab.slide_id == slide].to_csv(f'{pl}/cptac__{task}__{slide}.csv', index=False)
PYEOF
run() {  # cohort config task slide models_dir fold features save_code raw prod data intflag
  CUDA_VISIBLE_DEVICES=$GPU yes | $PY create_heatmaps_cancerscout.py --config_file "$2" \
      --process_list "$PL/$1__$3__$4.csv" --models_dir "$MODELS_CLAM/$5" --ckpt_path "$MODELS_CLAM/$5/s_$6_checkpoint.pt" \
      --save_exp_code "$8" --features_dir "$7" --raw_save_dir "$9" --production_save_dir "${10}" \
      --data_dir "${11}" --embed_dim 512 --model_name conch_v1 ${12:-}
}
if [ "$PANELS" != cptac ]; then
  need_cancerscout
  for f in "$PL"/cs__*.csv; do
    b=$(basename "$f" .csv); task=$(echo "$b" | cut -d_ -f3- | sed 's/__.*//'); slide=${b##*__}
    ds=$task; intl=""; [ "$task" = proteomic_classification ] && intl="--int_label_keys"
    fold=$(awk -F'\t' -v t="$task" -v s="$slide" '$1==t && $2==s {print $3}' "$PL/cs_folds.tsv")
    run cs config_cancerscout_examples.yaml "$task" "$slide" "cancerscout_colon_he_${ds}_v3_conch_s1" "$fold" \
        "$CSWSI_WORK/features/cancerscout/conch" "HEATMAP_pubfig_${task}_${slide}_fold${fold}" \
        "$CSWSI_WORK/heatmaps/raw_cancerscout" "$CSWSI_WORK/heatmaps/production_cancerscout" "$CSWSI_SLIDES_CS" "$intl"
  done
fi
need_cptac
for f in "$PL"/cptac__*.csv; do
  b=$(basename "$f" .csv); task=$(echo "$b" | cut -d_ -f3- | sed 's/__.*//'); slide=${b##*__}
  run cptac config_cptac_20x.yaml "$task" "$slide" "cancerscout_colon_he_${task}_v3_conch_s1" 8 \
      "$CSWSI_WORK/features/cptac/conch_scale20x_stainnet" "HEATMAP_cptaccohort20xsn_${task}_${slide}_fold8" \
      "$CSWSI_WORK/heatmaps/raw_cptac_20x" "$CSWSI_WORK/heatmaps/production_cptac_20x" "$CSWSI_SLIDES_CPTAC"
done
cd "$REPO/figures"
[ "$PANELS" = cptac ] || $PY fig5D_suppfig8_attention_grids.py
$PY suppfig9_cptac_attention_20x.py
