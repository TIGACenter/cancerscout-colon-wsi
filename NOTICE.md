# Third-party components

* **CLAM** (Lu, M. Y. et al. Data-efficient and weakly supervised computational pathology on
  whole-slide images. Nat. Biomed. Eng. 5, 555-570 (2021)), https://github.com/mahmoodlab/CLAM,
  GPL-3.0. `clam/` is a fork with study-specific entry points (`eval_cancerscout.py`,
  `create_heatmaps_cancerscout.py`, `create_splits_cancerscout.py`).
* **StainNet** architecture (Kang, H. et al. StainNet: a fast and robust stain normalization
  network. Front. Med. 8, 746307 (2021)), reimplemented in `stain_normalization/stainnet.py`.
* **Macenko** normalization (Macenko, M. et al. ISBI 2009), `stain_normalization/macenko.py`.
* **UNI** (Chen, R. J. et al. Nat. Med. 2024) and **CONCH** (Lu, M. Y. et al. Nat. Med. 2024)
  weights are not redistributed; they are obtained from MahmoodLab on Hugging Face under their own
  terms. The trained CLAM models on Zenodo contain no encoder weights; those built on UNI or CONCH
  features need the encoders to be used, under MahmoodLab's terms. The model weights are released under
  CC BY 4.0.
* **CPTAC-COAD** tissue slide images (The Cancer Imaging Archive, doi:10.7937/TCIA.YZWQ-ZZ63,
  CC BY 4.0) are not redistributed; `scripts/00_download_cptac.sh` fetches them from TCIA.
  The CPTAC labels in `configs/cptac_labels/` are derived from Vasaikar, S. et al. Proteogenomic
  analysis of human colon cancer reveals new therapeutic opportunities. Cell 177, 1035-1049 (2019).
