# Trained models (not released)

The trained models are not part of this release. The pipeline trains them (`scripts/04_train.sh`,
`scripts/06_stainnet.sh`) or reads them from here (or from `$CSWSI_MODELS`) to evaluate:

```
models/
  clam/cancerscout_colon_he_<dataset>_v3_<encoder>_s1/
      s_<fold>_checkpoint.pt          fold = 0..9, CLAM-MB weights
      split_<fold>_results.pkl        predictions on the fold's internal test split (decision thresholds)
  stainnet/stainnet_final.pt          StainNet, CPTAC -> discovery-cohort colors
```

`<encoder>` is `resnet`, `uni` or `conch`; `<dataset>` is the second column of
`configs/endpoints.tsv`. With these files in place, `scripts/run_cptac.sh` reproduces the CPTAC
results without the discovery-cohort data.
