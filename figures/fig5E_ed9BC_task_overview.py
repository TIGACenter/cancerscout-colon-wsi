#!/usr/bin/env python3
"""
Fig. 5E and Extended Data Fig. 9B,C: held-out benchmark per task and encoder.

Per-fold dots: AUC or balanced accuracy of each cross-validation model on the held-out set.
Diamonds: AUC of the probability averaged over the ten models (not the mean of the ten AUCs), or
balanced accuracy of the majority vote of the ten models' calls (each at its own threshold).
Fig. 5E shows the main endpoints; Extended Data Fig. 9B,C every endpoint. The three-class subgroup
task is the three CIN subgroups C2/C3/C5 among CIN cases. Values come from analysis/compute_numbers.py
(one row per slide). Print size: Fig. 5E 5.0 in wide, Extended Data Fig. 9B/C 6.9 in; fonts 7-8.5 pt.
"""
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper_style import (BASE, FULL_W, GRID, INK, INK2, METHOD_COLORS, METHODS, MUTED, TAB,
                         save, setup)

FIG5E = [('tumor_vs_normal', 'Tumor vs\nnormal'), ('msi_status_consensus', 'MSI status\n(consensus)'),
         ('braf', 'BRAF'), ('k_n_ras', 'KRAS/\nNRAS'), ('cptac', 'CPTAC\nsubtype'),
         ('proteomic_classification', 'Proteomic\nsubgroups\n(6-class)'),
         ('cin_vs_mcml', 'CIN vs\nmesen-\nchymal'),
         ('p_g_c2c3c5_cin_cases', 'CIN\nsubgroups\nC2/C3/C5')]
ED9 = [('Tumor / clinical', [('tumor_vs_normal', 'Tumor vs normal'), ('sex', 'Sex'), ('grading', 'Grade'),
                             ('ln_metastasis', 'Lymph-node metastasis'), ('distant_metastasis', 'Distant metastasis'),
                             ('cptac', 'CPTAC subtype')]),
       ('MSI', [('msi_status', 'MSI status (IHC)'), ('msi_status_genetic', 'MSI status (genetic)'),
                ('msi_status_consensus', 'MSI status (consensus)')]),
       ('Genetics', [('braf', 'BRAF'), ('k_n_ras', 'KRAS/NRAS'), ('tp53', 'TP53'), ('apc', 'APC')]),
       ('Proteomic', [('proteomic_classification', 'Subgroups C1-C6 (6-class)'),
                      ('p_g_c2c3c5_cin_cases', 'Subgroups C2/C3/C5 (CIN cases)'),
                      ('cin_vs_mcml', 'CIN vs mesenchymal')])]


def draw(ax, pairs, metric, ylabel, sections=None, legend_loc='lower center', show_cons=True, rotate=False):
    pf = pd.read_csv(f'{TAB}/in_domain_perfold.csv')
    cons = pd.read_csv(f'{TAB}/in_domain_metrics.csv')
    keys = [k for k, _ in pairs]
    lab = dict(pairs)
    order = [lab[k] for k in keys]
    d = pf[pf.task.isin(keys)].copy()
    d['task_label'] = d.task.map(lab)
    d['method_label'] = d.encoder.map(METHODS)
    hue = list(METHODS.values())
    sns.swarmplot(data=d, x='task_label', y=metric, hue='method_label', order=order, hue_order=hue,
                  palette=METHOD_COLORS, dodge=True, size=2.3, alpha=0.9, ax=ax, legend=False,
                  edgecolor='white', linewidth=0.25)
    ccol = {'auc': 'auc_consensus', 'bacc': 'bacc_consensus'}[metric]
    if show_cons:
        for j, k in enumerate(keys):
            for i, e in enumerate(METHODS):
                v = cons[(cons.task == k) & (cons.encoder == e)][ccol]
                if len(v) and np.isfinite(v.values[0]):
                    ax.scatter(j + (i - 1) * 0.8 / 3, v.values[0], marker='D', s=11, facecolor=INK,
                               edgecolor='white', linewidth=0.4, zorder=7)
    n = len(keys)
    ax.axhline(0.5, color=MUTED, ls=(0, (1, 1.5)), lw=0.6, zorder=1)
    ax.text(n - 0.45, 0.5, 'chance', color=MUTED, fontsize=6.5, va='center', ha='left', clip_on=False)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', length=0, pad=2, labelsize=6.6)
    if rotate:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha='right', rotation_mode='anchor')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    if sections:
        start = 0
        tr = ax.get_xaxis_transform()
        for title, items in sections:
            if start:
                ax.axvline(start - 0.5, color=BASE, lw=0.6, zorder=1)
            ax.text(start + (len(items) - 1) / 2, 1.02, title, transform=tr, ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color=INK2)
            start += len(items)
    handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], label=m) for m in hue]
    diamond = {'auc': 'Averaged prediction (10 models)', 'bacc': 'Majority vote (10 models)'}[metric]
    handles.append(mlines.Line2D([], [], color=INK, marker='D', ls='None', markeredgecolor='white',
                                 markeredgewidth=0.4, markersize=4, label=diamond))
    # one horizontal row in the empty band near zero: no data point falls below 0.35
    leg = ax.legend(handles=handles, loc=legend_loc, ncol=4, frameon=True, framealpha=0.95,
                    edgecolor=BASE, handlelength=0.9, borderpad=0.35, columnspacing=1.0,
                    handletextpad=0.4, bbox_to_anchor=(0.5, 0.03))
    leg.get_frame().set_linewidth(0.5)


def main():
    setup()
    fig, ax = plt.subplots(figsize=(5.9, 2.55))
    draw(ax, FIG5E, 'auc', 'AUC (held-out set)')
    save(fig, 'Fig5E_heldout_AUC')

    ed_pairs = [p for _, items in ED9 for p in items]
    for metric, ylabel, name in (('auc', 'AUC (held-out set)', 'ExtDataFig9B_heldout_AUC'),
                                 ('bacc', 'Balanced accuracy\n(held-out set)', 'ExtDataFig9C_heldout_balanced_accuracy')):
        fig, ax = plt.subplots(figsize=(FULL_W, 2.3))
        draw(ax, ed_pairs, metric, ylabel, sections=ED9, rotate=True)
        save(fig, name)


if __name__ == '__main__':
    main()
