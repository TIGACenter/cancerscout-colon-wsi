# Data

| Folder | Cohort | Status |
|---|---|---|
| `cancerscout/` | Discovery cohort (CancerScout colon, H&E) | Not public yet; released with the paper (BioImage Archive). Empty until then. |
| `cptac/` | External cohort (CPTAC-COAD, H&E) | Public (TCIA). `scripts/00_download_cptac.sh` fills `cptac/slides/`. |

Nothing in these folders except the READMEs is tracked by git. To keep the data elsewhere, set the
paths in `configs/paths.env` (template: `configs/paths.env.example`).
