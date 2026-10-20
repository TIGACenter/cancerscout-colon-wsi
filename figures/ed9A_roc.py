#!/usr/bin/env python3
"""
Extended Data Fig. 9A: ROC curves on the held-out set, one curve per encoder.

The solid curve and the legend AUC use the probability averaged over the ten models (the value the
diamonds in Fig. 5E show); the band is +-1 SD of the ten per-fold curves. Multiclass curves are
macro-averaged one-vs-rest, matching the AUC in Fig. 5E. One row per slide; one row of four panels at
6.9 in; fonts 7-8.5 pt.
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from paper_style import EVALD, FULL_W, GRID, METHOD_COLORS, METHODS, MUTED, save, setup

TASKS = [('tumor_vs_normal', 'Tumor detection'), ('msi_status_consensus', 'MSI status (consensus)'),
         ('cptac', 'CPTAC subtype'), ('p_g_c2c3c5_cin_cases', 'CIN subgroups C2/C3/C5\n(CIN cases)')]
GRID_FPR = np.linspace(0, 1, 201)


def roc_macro(y, P):
    """Binary: ROC of the positive class. Multiclass: macro average of one-vs-rest curves."""
    if P.shape[1] == 2:
        f, t, _ = roc_curve(y, P[:, 1])
        return np.interp(GRID_FPR, f, t), roc_auc_score(y, P[:, 1])
    ts, aucs = [], []
    for c in range(P.shape[1]):
        if not (y == c).any():
            continue
        f, t, _ = roc_curve((y == c).astype(int), P[:, c])
        ts.append(np.interp(GRID_FPR, f, t))
        aucs.append(roc_auc_score((y == c).astype(int), P[:, c]))
    return np.mean(ts, 0), float(np.mean(aucs))


def main():
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.05), sharey=True)
    for ax, (task, title) in zip(axes, TASKS):
        ax.plot([0, 1], [0, 1], ls=(0, (2, 2)), color=MUTED, lw=0.7)
        for enc, name in METHODS.items():
            fs = sorted(glob.glob(f'{EVALD}/EVAL_cancerscout_colon_he_{task}_v3_{enc}_s1/fold_*.csv'))
            folds = [pd.read_csv(f).drop_duplicates('slide_id') for f in fs]
            pc = sorted([c for c in folds[0].columns if c.startswith('p_')], key=lambda c: int(c[2:]))
            per = np.vstack([roc_macro(f.Y.astype(int).values, f[pc].values)[0] for f in folds])
            per[:, 0] = 0
            cons = pd.concat(folds).groupby('slide_id')[pc + ['Y']].mean()
            tpr, a = roc_macro(cons.Y.round().astype(int).values, cons[pc].values)
            tpr[0] = 0
            col = METHOD_COLORS[name]
            ax.fill_between(GRID_FPR, np.clip(per.mean(0) - per.std(0), 0, 1),
                            np.clip(per.mean(0) + per.std(0), 0, 1), color=col, alpha=0.18, lw=0)
            ax.plot(GRID_FPR, tpr, color=col, lw=1.1, label=f'{name}  {a:.2f}')
        ax.set_title(title, fontsize=7.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.set_xticklabels(['0', '0.5', '1']); ax.set_yticklabels(['0', '0.5', '1'])
        ax.set_xlabel('1 - specificity')
        ax.grid(color=GRID, lw=0.5)
        ax.set_aspect('equal')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        leg = ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='none',
                        fontsize=6.3, handlelength=1.0, borderpad=0.3, labelspacing=0.2,
                        title='AUC, averaged prediction', title_fontsize=6.3)
    axes[0].set_ylabel('Sensitivity')
    fig.tight_layout(w_pad=1.0)
    save(fig, 'ExtDataFig9A_heldout_ROC')


if __name__ == '__main__':
    main()
