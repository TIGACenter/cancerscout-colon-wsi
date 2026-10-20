#!/usr/bin/env python3
"""
CPTAC at matched 20x: balanced accuracy of the Fig. 5F endpoints raw vs StainNet (Fig. 5F style),
and whether normalization moves each model's best CPTAC threshold toward its CancerScout threshold.

(a, b) balanced accuracy per model at its CancerScout threshold (dots) and of the ten-model majority
       vote (diamonds); CPTAC subtype by argmax; chance 0.5 (MSI) and 1/3 (subtype).
(c)    threshold shift per model: logit(best CPTAC threshold) - logit(CancerScout threshold);
       0 = the transferred threshold is already the best one; raw (open) -> StainNet (filled).
(d)    balanced accuracy lost by using the CancerScout threshold instead of the best CPTAC one.
Data: tables/cptac_perfold.csv, cptac_metrics.csv, cptac_threshold_migration_20x.csv.
"""
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper_style import BASE, FULL_W, GRID, INK, INK2, METHOD_COLORS, METHODS, MUTED, TAB, save, setup

TASKS = [('msi_status', 'MSI\n(IHC)'), ('msi_status_genetic', 'MSI\n(genetic)'),
         ('msi_status_consensus', 'MSI\n(consensus)'), ('cptac', 'CPTAC\nsubtype')]
COND = {'20x raw': 'Raw', '20x StainNet': 'StainNet'}


def bacc_panel(ax, pf, cons, condition, title):
    keys = [k for k, _ in TASKS]
    lab = dict(TASKS)
    d = pf[(pf.condition == condition) & pf.task.isin(keys)].copy()
    d['task_label'] = d.task.map(lab)
    d['method_label'] = d.encoder.map(METHODS)
    sns.swarmplot(data=d, x='task_label', y='bacc', hue='method_label', order=[lab[k] for k in keys],
                  hue_order=list(METHODS.values()), palette=METHOD_COLORS, dodge=True, size=2.4, alpha=0.9,
                  ax=ax, legend=False, edgecolor='white', linewidth=0.25)
    for j, k in enumerate(keys):
        for i, e in enumerate(METHODS):
            v = cons[(cons.task == k) & (cons.encoder == e) & (cons.condition == condition)].bacc_consensus
            ax.scatter(j + (i - 1) * 0.8 / 3, v.values[0], marker='D', s=12, facecolor=INK,
                       edgecolor='white', linewidth=0.4, zorder=7)
        chance = 1 / 3 if k == 'cptac' else 0.5
        ax.plot([j - 0.45, j + 0.45], [chance, chance], color=MUTED, ls=(0, (1, 1.5)), lw=0.7, zorder=1)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_xlabel('')
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold')
    ax.tick_params(axis='x', length=0, pad=2, labelsize=6.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def paired_panel(ax, m, col, ylabel, zero_label):
    keys = [k for k, _ in TASKS[:3]]
    xt, xl = [], []
    rng = np.random.default_rng(1)
    for j, k in enumerate(keys):
        for i, e in enumerate(METHODS):
            x0 = j * 3.6 + i * 1.1
            col_e = METHOD_COLORS[METHODS[e]]
            r = m[(m.task == k) & (m.encoder == e) & (m.condition == 'raw')].sort_values('fold')[col].values
            s = m[(m.task == k) & (m.encoder == e) & (m.condition == 'StainNet')].sort_values('fold')[col].values
            jit = rng.uniform(-0.08, 0.08, len(r))
            for a, b, jj in zip(r, s, jit):
                ax.plot([x0 + jj, x0 + 0.5 + jj], [a, b], color=col_e, lw=0.4, alpha=0.45)
            ax.scatter(x0 + jit, r, s=7, facecolor='white', edgecolor=col_e, lw=0.7, zorder=3)
            ax.scatter(x0 + 0.5 + jit, s, s=7, facecolor=col_e, edgecolor=col_e, lw=0.7, zorder=3)
            for xx, v in ((x0, r), (x0 + 0.5, s)):
                ax.plot([xx - 0.17, xx + 0.17], [np.median(v)] * 2, color=INK, lw=1.1, zorder=4)
        xt.append(j * 3.6 + 1.35)
        xl.append(dict(TASKS)[k].replace('\n', ' '))
        if j:
            ax.axvline(j * 3.6 - 0.55, color=BASE, lw=0.6)
    ax.axhline(0, color=MUTED, ls=(0, (1, 1.5)), lw=0.7)
    ax.text(10.35, 0, f' {zero_label}', color=MUTED, fontsize=6.3, va='center', ha='left')
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, 10.3)
    ax.set_ylabel(ylabel)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def main():
    setup()
    pf = pd.read_csv(f'{TAB}/cptac_perfold.csv')
    cons = pd.read_csv(f'{TAB}/cptac_metrics.csv')
    m = pd.read_csv(f'{TAB}/cptac_threshold_migration_20x.csv')
    fig = plt.figure(figsize=(FULL_W, 7.3))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.62, wspace=0.12)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1], sharey=axa)
    bacc_panel(axa, pf, cons, '20x raw', 'a  Balanced accuracy, CPTAC 20x, raw')
    bacc_panel(axb, pf, cons, '20x StainNet', 'b  Balanced accuracy, CPTAC 20x, StainNet')
    axa.set_ylabel('Balanced accuracy\n(CancerScout thresholds)')
    axb.set_ylabel('')
    plt.setp(axb.get_yticklabels(), visible=False)
    handles = [mpatches.Patch(facecolor=METHOD_COLORS[v], label=v) for v in METHODS.values()]
    handles.append(mlines.Line2D([], [], color=INK, marker='D', ls='None', markeredgecolor='white',
                                 markeredgewidth=0.4, markersize=4, label='Majority vote (10 models)'))
    axb.legend(handles=handles, loc='lower right', ncol=2, fontsize=6.4, frameon=True, framealpha=0.95,
               edgecolor=BASE, handlelength=0.9, columnspacing=0.8, handletextpad=0.4)
    axc = fig.add_subplot(gs[1, :])
    paired_panel(axc, m, 'shift', 'Threshold shift\n(logit)',
                 'no shift')
    axc.set_title('c  Threshold shift: best CPTAC threshold minus CancerScout threshold', loc='left',
                  fontsize=8.5, fontweight='bold')
    axc.text(0.0, -0.2, 'Open circles: raw; filled: StainNet; lines join the same model; bars: medians.\n'
             'Below 0: the transferred threshold is too high, so the model under-calls MSI on CPTAC; '
             'above 0: it over-calls MSI.', transform=axc.transAxes, fontsize=6.3, color=INK2, va='top')
    axd = fig.add_subplot(gs[2, :])
    paired_panel(axd, m, 'lost', 'Balanced accuracy\nlost', 'none lost')
    axd.set_title('d  Balanced accuracy lost by using the CancerScout threshold instead of the best CPTAC one',
                  loc='left', fontsize=8.5, fontweight='bold')
    save(fig, 'CPTAC_20x_balanced_accuracy_and_threshold_shift')


def ed10_cd():
    """Balanced accuracy raw and StainNet, as Extended Data Fig. 10C,D (the threshold-shift panel is
    analysis only; the authors left it out of the paper)."""
    setup()
    pf = pd.read_csv(f'{TAB}/cptac_perfold.csv')
    cons = pd.read_csv(f'{TAB}/cptac_metrics.csv')
    fig, (axc, axd) = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True, gridspec_kw={'wspace': 0.12})
    bacc_panel(axc, pf, cons, '20x raw', 'C  Balanced accuracy, raw')
    bacc_panel(axd, pf, cons, '20x StainNet', 'D  Balanced accuracy, StainNet')
    axc.set_ylabel('Balanced accuracy on CPTAC\n(CancerScout thresholds)')
    axd.set_ylabel('')
    plt.setp(axd.get_yticklabels(), visible=False)
    handles = [mpatches.Patch(facecolor=METHOD_COLORS[v], label=v) for v in METHODS.values()]
    handles.append(mlines.Line2D([], [], color=INK, marker='D', ls='None', markeredgecolor='white',
                                 markeredgewidth=0.4, markersize=4, label='Majority vote (10 models)'))
    axd.legend(handles=handles, loc='lower right', ncol=2, fontsize=6.4, frameon=True, framealpha=0.95,
               edgecolor=BASE, handlelength=0.9, columnspacing=0.8, handletextpad=0.4)
    save(fig, 'ExtDataFig10CD_CPTAC_balanced_accuracy_20x')


if __name__ == '__main__':
    main()
    ed10_cd()
