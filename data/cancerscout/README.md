# Discovery cohort (CancerScout colon): placeholder

The H&E whole-slide images and slide-level labels of the discovery cohort are deposited in the
BioImage Archive and will be released upon peer-reviewed publication of the paper (accession
in the paper's Data availability statement). This folder stays empty until then.

Layout the pipeline expects once the data are released:

```
data/cancerscout/
  slides/                        <slide_id>.tiff   (2,496 H&E WSIs used in the benchmarks)
  labels/<dataset>/train.csv     case_id,slide_id,label   (70% of patients: cross-validation)
  labels/<dataset>/eval.csv      case_id,slide_id,label   (30% of patients: held-out evaluation set)
  labels/annotations/cptac_subtype_per_slide.csv          slide_id,cptac_subtype   (Extended Data Fig. 9D)
  stain_reference/*.png          the three 256 px discovery-cohort tiles used as the StainNet target
```

`<dataset>` is the second column of `configs/endpoints.tsv`. The split is patient-level: no patient
has slides on both sides.
