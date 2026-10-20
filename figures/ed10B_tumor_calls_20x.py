#!/usr/bin/env python3
"""
Extended Data Fig. 10B: tumor calls on the 372 tumor-positive CPTAC slides at matched 20x, raw
versus StainNet.

Each slide is called by each of the ten models at that model's discovery-cohort threshold, and the
slide call is the majority vote (5-5 ties broken by the averaged probability). The panel shows, per
slide, how many of the ten models call it tumor; the percentage is the share of slides called tumor
by the majority vote.
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))
from consensus_vote import majority_vote, per_model_calls  # noqa: E402
from paper_style import EVALD, FULL_W, GRID, INK, METHODS, MUTED, save, setup  # noqa: E402

COND = [('scale20x_full', 'Raw'), ('scale20xsn_full', 'StainNet')]
CCOL = {'Raw': '#c8791b', 'StainNet': '#3f51b5'}


def tumor_votes(enc, suf):
    s = pd.read_csv(f'{EVALD}/EVAL_cancerscout_colon_he_tumor_vs_normal_v3_{enc}_s1/summary_best_th_cons.csv')
    th = list(s[s.folds.astype(str) != 'consensus'].test_threshold_best.astype(float))
    fs = [pd.read_csv(f).drop_duplicates('slide_id') for f in
          sorted(glob.glob(f'{EVALD}/EVAL_cancerscout_colon-cptac_tumor_vs_normal_v3_{enc}_{suf}/fold_*.csv'),
                 key=lambda p: int(p.split('_')[-1][:-4]))]
    n_tumor = (per_model_calls(fs, th) == 0).sum(1)        # class 0 = tumor
    vote, _ = majority_vote(fs, th)
    return n_tumor, (vote == 0).loc[n_tumor.index]


def main():
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (enc, name) in zip(axes, METHODS.items()):
        for i, (suf, lab) in enumerate(COND):
            n, called = tumor_votes(enc, suf)
            v = n.values.astype(float)
            parts = ax.violinplot(v, positions=[i], widths=0.75, showextrema=False, bw_method=0.35)
            for b in parts['bodies']:
                b.set_facecolor(CCOL[lab]); b.set_alpha(0.3); b.set_edgecolor('none')
            ax.scatter(i + (rng.random(len(v)) - 0.5) * 0.3, v + (rng.random(len(v)) - 0.5) * 0.5, s=1.6,
                       color=CCOL[lab], alpha=0.55, lw=0)
            ax.scatter([i], [v.mean()], marker='D', s=14, facecolor=INK, edgecolor='white', lw=0.5, zorder=6)
            ax.text(i, 10.9, f'{100 * called.mean():.1f}%', ha='center', fontsize=7.5, fontweight='bold',
                    color=CCOL[lab])
        ax.axhspan(4.55, 5.45, color='#ecebe4', lw=0, zorder=0)     # 5 of 10: tie
        ax.axhline(5.5, ls=(0, (1, 1.5)), lw=0.6, color=MUTED)
        ax.set_xticks(range(len(COND)))
        ax.set_xticklabels([c for _, c in COND])
        ax.set_xlim(-0.6, len(COND) - 0.4)
        ax.set_title(name, fontsize=8.5, fontweight='bold', pad=11)
        ax.set_ylim(-0.6, 11.4)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Models calling the slide\ntumor (of 10)')
    # annotations in the UNI panel, whose lower half is empty
    axes[1].text(0.5, 5.6, 'above: majority calls tumor', fontsize=6.3, color=MUTED, va='bottom', ha='center')
    axes[1].text(0.5, 4.4, '5 of 10 (band): tie, decided\nby the averaged probability', fontsize=6.3,
                 color=MUTED, va='top', ha='center')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'ExtDataFig10B_CPTAC_tumor_calls_20x')


if __name__ == '__main__':
    main()
