#!/usr/bin/env python3
"""
Extended Data Fig. 9D: proteomic-subgroup composition of the CPTAC subtypes on the held-out
discovery slides, proteome-derived reference versus the slide-level predictions of the six-class
models (ResNet50, UNI, CONCH).

Slide-level predictions are the majority vote of the ten cross-validation models
(analysis/consensus_vote.py). The CPTAC subtype of each slide is the per-slide annotation released
with the discovery cohort (<labels>/annotations/cptac_subtype_per_slide.csv); slides without a CIN,
MSI or mesenchymal subtype are not shown. Stacking and legend in C1-C6 order; print size 6.9 in wide,
fonts 7-8.5 pt; percentages printed from 15%.
"""
import glob
import json
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))), 'analysis'))
from consensus_vote import majority_vote  # noqa: E402
from paper_style import (EVALD, FULL_W, GRID, METHODS, SUBGROUP_COLORS, SUBGROUPS, save, setup)

ORDER = ['1', '2', '3', '4', '5', '6']
ROWS = ['CIN', 'MSI', 'Mesenchymal']


def main():
    setup()
    parts = []
    for enc in METHODS:
        d = f'{EVALD}/EVAL_cancerscout_colon_he_proteomic_classification_v3_{enc}_s1'
        inv = {v: k for k, v in json.load(open(f'{d}/label_dict.json')).items()}
        folds = [pd.read_csv(f).drop_duplicates('slide_id') for f in
                 sorted(glob.glob(f'{d}/fold_*.csv'), key=lambda p: int(p.split('_')[-1][:-4]))]
        vote, _ = majority_vote(folds)
        y = folds[0].set_index('slide_id').Y.astype(int)
        parts.append(pd.DataFrame({'slide_id': vote.index, 'method': enc,
                                   'Y': y.loc[vote.index].map(inv).values,
                                   'consensus_pred': vote.map(inv).values}))
    w = pd.concat(parts, ignore_index=True)
    import cswsi_config as cfg
    # per-slide CPTAC subtype annotation of the discovery cohort (CIN, MSI, Mesenchymal, Other)
    lab = pd.read_csv(f'{cfg.LABELS}/annotations/cptac_subtype_per_slide.csv')
    cmap = lab.dropna(subset=['cptac_subtype']).drop_duplicates('slide_id').set_index('slide_id').cptac_subtype
    w['cptac'] = w.slide_id.map(cmap)
    w = w.dropna(subset=['cptac'])
    cols = [('Proteome-derived reference\n(ground truth)', None)] + [(m, k) for k, m in METHODS.items()]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.75), sharey=True)
    for ax, (title, key) in zip(axes, cols):
        d = w[w.method == 'uni'] if key is None else w[w.method == key]
        col = 'Y' if key is None else 'consensus_pred'
        ct = pd.crosstab(d.cptac, d[col]).reindex(index=ROWS, columns=ORDER, fill_value=0)
        frac = ct.div(ct.sum(1), axis=0)
        for yi, r in enumerate(ROWS[::-1]):
            left = 0.0
            for g in ORDER:
                v = float(frac.loc[r, g])
                if v <= 0:
                    continue
                ax.barh(yi, v, left=left, height=0.62, color=SUBGROUP_COLORS[g], edgecolor='white', lw=0.4)
                if v >= 0.15:
                    ax.text(left + v / 2, yi, f'{v:.0%}', ha='center', va='center', fontsize=6.3,
                            color='white' if g in ('1', '2', '5', '6') else 'black', fontweight='bold')
                left += v
        n = ct.sum(1)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([f'{r}\n(n = {n[r]})' for r in ROWS[::-1]])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .5, 1])
        ax.set_xticklabels(['0', '50', '100'])
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('CPTAC subtype')
    handles = [mpatches.Patch(color=SUBGROUP_COLORS[g], label=SUBGROUPS[g]) for g in ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.55, -0.1), handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    fig.supxlabel('Share of slides (%)', fontsize=7.5, y=0.1)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'ExtDataFig9D_subgroup_composition')


if __name__ == '__main__':
    main()
