# Trained models

The trained models of the paper are archived on Zenodo (record and DOI in `configs/zenodo.env`) and
downloaded into this directory (or `$CSWSI_MODELS`) by

```bash
bash scripts/download_models.sh                    # all three encoders and StainNet, about 1.2 GB
ENCODERS="conch" bash scripts/download_models.sh   # one encoder only (plus StainNet)
```

The script checks every archive against `configs/model_archives.sha256` before it unpacks it, and
keeps the archives in `models/.archives/`, so a second run does not download again. Without network
access, copy the archives into a directory and run `MODEL_ARCHIVES=<dir> bash scripts/download_models.sh`.
`scripts/run_cptac.sh` runs this step first. The record page can also be opened in a browser and the
archives unpacked here by hand (`unzip -o <archive> -d models/`).

## Archives

| Archive | Content | Size |
|---|---|---|
| `cswsi_models_clam_resnet.zip` | CLAM-MB on ResNet50 (ImageNet) tile features: 16 endpoints x 10 models | 453 MB |
| `cswsi_models_clam_uni.zip` | the same on UNI tile features | 469 MB |
| `cswsi_models_clam_conch.zip` | the same on CONCH tile features | 315 MB |
| `cswsi_models_stainnet.zip` | StainNet, CPTAC -> discovery-cohort colors | 6 KB |

## Layout after download

```
models/
  clam/cancerscout_colon_he_<dataset>_v3_<encoder>_s1/
      s_<fold>_checkpoint.pt          fold = 0..9, CLAM-MB weights (PyTorch state dict)
      label_dict.json                 class name -> output index
  clam/thresholds_<encoder>.csv       decision threshold of every binary model (dataset, encoder, fold, threshold)
  stainnet/stainnet_final.pt          StainNet, CPTAC -> discovery-cohort colors
```

`<encoder>` is `resnet`, `uni` or `conch`; `<dataset>` is the second column of
`configs/endpoints.tsv`. The models take tile embeddings, not images: the UNI and CONCH models need
the UNI and CONCH encoders, which MahmoodLab distributes on Hugging Face under their own gated
licenses (main README, Encoder weights).

The thresholds were fitted by `analysis/validation_thresholds.py` on each model's internal test split
of the training set (Youden's J), never on the held-out set. The release ships them as a table
instead of CLAM's per-split prediction files (`split_<fold>_results.pkl`), which list
discovery-cohort slides; `analysis/validation_thresholds.py` reads the table when those files are
absent. A model trained here with `scripts/04_train.sh` writes the prediction files, and the same
script then fits its thresholds from them.

What each model predicts, its classes, training data and limits: `docs/MODELS.md`.

## License

The model weights are released under CC BY 4.0; the code in this repository stays under GPL-3.0.
Using the UNI and CONCH models requires the UNI and CONCH encoders, which remain under MahmoodLab's
own terms.
