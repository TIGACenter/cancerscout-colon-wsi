# External cohort: CPTAC-COAD

All 372 H&E tissue slide images of the TCIA collection CPTAC-COAD (178 patients, all
tumor-positive), doi:10.7937/TCIA.YZWQ-ZZ63, license CC BY 4.0.

```bash
gem install aspera-cli && ascli config transferd install   # Aspera CLI, once
bash scripts/00_download_cptac.sh                          # 66.7 GB into data/cptac/slides/, then verified
```

TCIA distributes these slides only as an Aspera Faspex package (link on the collection page,
https://www.cancerimagingarchive.net/collection/cptac-coad/, "Tissue Slide Images"). If the Aspera
CLI cannot be used, download the package in a browser, put the `.svs` files in `data/cptac/slides/`
and run `VERIFY_ONLY=1 bash scripts/00_download_cptac.sh`. Every file is checked against
`configs/cptac_manifest.tsv` (name, size, MD5 of the copy analysed in the paper).

Labels (`configs/cptac_labels/`, one row per slide): MSI status from the sequencing-based call
(`MSmutectFishercall`, 223 slides from 106 patients) and the UMS subtype (190 slides from 87
patients) of Vasaikar et al., Cell 2019; tumor detection uses all 372 slides.

Please cite the collection when using these data (full citation and the required CPTAC
acknowledgment on the collection page): The Clinical Proteomic Tumor Analysis Consortium Colon
Adenocarcinoma Collection (CPTAC-COAD), The Cancer Imaging Archive,
doi:10.7937/TCIA.YZWQ-ZZ63.
