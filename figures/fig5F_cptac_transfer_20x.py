#!/usr/bin/env python3
"""
Fig. 5F: transfer to external CPTAC slides at matched 20x after StainNet normalization.

CPTAC tiles at the training field of view (512 px level-0 crops = 20x, ~128 um; cptac/reextract_20x.py,
condition 'scale20x_stainnet'), all CPTAC slides with a label (223 slides / 106 cases for MSI, 190 / 87
for CPTAC subtype). Dots: AUC of each cross-validation model; diamonds: AUC of the probability averaged
over the ten models. The three MSI columns are the three discovery-cohort models (trained on the IHC,
genetic and consensus MSI labels), all scored against the single CPTAC MSI label. Print size 3.3 in.
"""
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from paper_style import BASE, GRID, INK, METHOD_COLORS, METHODS, MUTED, TAB, save, setup

TASKS = [('msi_status', 'MSI\n(IHC)'), ('msi_status_genetic', 'MSI\n(genetic)'),
         ('msi_status_consensus', 'MSI\n(consensus)'), ('cptac', 'CPTAC\nsubtype')]


def main(condition='20x StainNet', name='Fig5F_CPTAC_20x_StainNet_AUC'):
    setup()
    pf = pd.read_csv(f'{TAB}/cptac_perfold.csv')
    cons = pd.read_csv(f'{TAB}/cptac_metrics.csv')
    keys = [k for k, _ in TASKS]
    lab = dict(TASKS)
    d = pf[(pf.condition == condition) & pf.task.isin(keys)].copy()
    d['task_label'] = d.task.map(lab)
    d['method_label'] = d.encoder.map(METHODS)
    hue = list(METHODS.values())
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    sns.swarmplot(data=d, x='task_label', y='auc', hue='method_label', order=[lab[k] for k in keys],
                  hue_order=hue, palette=METHOD_COLORS, dodge=True, size=2.4, alpha=0.9, ax=ax,
                  legend=False, edgecolor='white', linewidth=0.25)
    for j, k in enumerate(keys):
        for i, e in enumerate(METHODS):
            v = cons[(cons.task == k) & (cons.encoder == e) & (cons.condition == condition)].auc_consensus
            ax.scatter(j + (i - 1) * 0.8 / 3, v.values[0], marker='D', s=12, facecolor=INK,
                       edgecolor='white', linewidth=0.4, zorder=7)
    ax.axhline(0.5, color=MUTED, ls=(0, (1, 1.5)), lw=0.6, zorder=1)
    ax.text(len(keys) - 0.5, 0.51, 'chance', color=MUTED, fontsize=6.5, va='bottom', ha='right')
    ax.set_ylim(0, 1.04)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_xlabel('')
    ax.set_ylabel('AUC (CPTAC, 20x, StainNet)')
    ax.tick_params(axis='x', length=0, pad=2, labelsize=6.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], label=m) for m in hue]
    handles.append(mlines.Line2D([], [], color=INK, marker='D', ls='None', markeredgecolor='white',
                                 markeredgewidth=0.4, markersize=4, label='Averaged prediction\n(10 models)'))
    leg = ax.legend(handles=handles, loc='lower center', ncol=2, frameon=True, framealpha=0.95,
                    edgecolor=BASE, handlelength=0.9, borderpad=0.35, columnspacing=0.8,
                    handletextpad=0.4, bbox_to_anchor=(0.5, 0.02), fontsize=6.8)
    leg.get_frame().set_linewidth(0.5)
    save(fig, name)


if __name__ == '__main__':
    main()
