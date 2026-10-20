#!/usr/bin/env python3
"""
Supplementary Fig. 9: attention maps of the CONCH models on five CPTAC slides (two CIN, two MSI,
one mesenchymal) at matched 20x after StainNet normalization.

Blockmaps from scripts/11_heatmaps.sh (20x StainNet features, 512 px tiles, one pre-specified model
per task: fold 8). The predicted label above each map is the majority vote of the ten models (MSI:
each at its own threshold; CPTAC subtype: each model's argmax), not the single model's call and not
a CPTAC-calibrated threshold. Example slides: configs/example_slides.yaml. Print size 6.9 in;
fonts 7-7.5 pt.
"""
import glob

import sys

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image

sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))), 'analysis'))
from consensus_vote import majority_vote, per_model_calls  # noqa: E402
from paper_style import EVALD, FULL_W, save, setup  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
import cswsi_config as cfg  # noqa: E402
RAW = cfg.HEATMAPS + '/raw_cptac_20x'
# five cases, as many as the held-out examples of Supplementary Fig. 8: two CIN, two MSI, one
# mesenchymal (the current figure's third slide was a second section of case 05CO033)
SLIDES = ['09CO019-b913f361-c5c2-4e49-91ca-719bd8', '01CO015-b873f54a-5a86-4dca-90f9-715820',
          '05CO041-ce47b803-bd11-473f-b4a2-f99643', '11CO036-2739ad00-dcb2-4485-ac9b-b77851',
          '05CO033-a4ae882e-5378-4acb-bd16-ddd4e8']
TASKS = [('tumor_vs_normal', 'Tumor detection'), ('cptac', 'CPTAC subtype'), ('msi_status', 'MSI status')]


def consensus(task):
    d = f'{EVALD}/EVAL_cancerscout_colon-cptac_{task}_v3_conch_scale20xsn_full'
    import json
    inv = {v: k for k, v in json.load(open(f'{d}/label_dict.json')).items()}
    folds = [pd.read_csv(f).drop_duplicates('slide_id').set_index('slide_id')
             for f in sorted(glob.glob(f'{d}/fold_*.csv'), key=lambda p: int(p.split('_')[-1][:-4]))]
    pc = sorted([c for c in folds[0].columns if c.startswith('p_')], key=lambda c: int(c[2:]))
    mean = pd.concat(folds).groupby(level=0)[pc + ['Y']].mean()
    th = None
    if task in ('msi_status', 'tumor_vs_normal'):
        t = pd.read_csv(f'{EVALD}/EVAL_cancerscout_colon_he_{task}_v3_conch_s1/summary_best_th_cons.csv')
        th = list(t[t.folds.astype(str) != 'consensus'].test_threshold_best.astype(float))
    pred, _ = majority_vote([f.reset_index() for f in folds], th)
    # number of the ten models agreeing with the majority call, for the panel title
    agree = (per_model_calls([f.reset_index() for f in folds], th).values == pred.values[:, None]).sum(1)
    mean['agree'] = pd.Series(agree, index=pred.index)
    return mean, pred.map(inv), mean.Y.round().astype(int).map(inv)


def blockmap(task, slide):
    hits = glob.glob(f'{RAW}/HEATMAP_cptaccohort20xsn_{task}_{slide}_fold8/*/{slide}/{slide}_blockmap.png')
    assert hits, (task, slide)
    return Image.open(hits[0])


def main():
    setup()
    C = {t: consensus(t) for t, _ in TASKS}
    fig, axes = plt.subplots(len(SLIDES), 4, figsize=(FULL_W, 1.95 * len(SLIDES)))
    for r, slide in enumerate(SLIDES):
        w = openslide.OpenSlide(f'{cfg.SLIDES_CPTAC}/{slide}.svs')
        lvl = min(5, w.level_count - 1)
        axes[r, 0].imshow(w.read_region((0, 0), lvl, w.level_dimensions[lvl]).convert('RGB'))
        axes[r, 0].set_title(f'CPTAC case {slide.split("-")[0]}\nH&E\n', fontsize=7.2, linespacing=1.1)
        for c, (task, name) in enumerate(TASKS, start=1):
            axes[r, c].imshow(blockmap(task, slide))
            mean, pred, true = C[task]
            lab = {'positive': 'tumor', 'negative': 'tumor-free'}.get(pred[slide], pred[slide])
            truth = {'positive': 'tumor'}.get(true[slide], true[slide])
            txt = (f'{name}\ntrue: {truth}\npredicted: {lab} '
                   f'({int(mean.loc[slide, "agree"])} of 10)')
            axes[r, c].set_title(txt, fontsize=7.2, linespacing=1.1)
        for ax in axes[r]:
            ax.axis('off')
    fig.tight_layout(h_pad=0.8, w_pad=0.3)
    save(fig, 'SuppFig9_CPTAC_attention_20x')
    # the other seven slides that were rendered, for the authors to choose from
    rows = []
    for slide in sorted({p.split('_fold8')[0].split('tumor_vs_normal_')[-1]
                         for p in glob.glob(f'{RAW}/HEATMAP_cptaccohort20xsn_tumor_vs_normal_*')}):
        rows.append(dict(slide=slide, **{f'{t}_true': C[t][2].get(slide) for t, _ in TASKS[1:]},
                         **{f'{t}_pred': C[t][1].get(slide) for t, _ in TASKS[1:]},
                         tumor_pred=C['tumor_vs_normal'][1].get(slide)))
    pd.DataFrame(rows).to_csv(f'{cfg.TABLES}/suppfig9_rendered_slides.csv', index=False)


if __name__ == '__main__':
    main()
