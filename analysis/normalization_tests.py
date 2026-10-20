#!/usr/bin/env python3
"""
Does StainNet normalization change CPTAC transfer at the matched 20x resolution?

For each endpoint and encoder, raw 20x and StainNet 20x are compared on the same slides with
the same ten frozen models, so every comparison is paired.
  * AUC endpoints: difference of the consensus AUC (StainNet - raw) with a case-level paired
    bootstrap (2,000 resamples of CPTAC cases, all slides of a case kept together): 95% CI and
    two-sided p; plus a Wilcoxon signed-rank test over the ten per-fold AUCs.
  * Tumor detection (every CPTAC slide is tumor, so no AUC exists): share of slides called
    tumor at P > 0.5 (exact McNemar test on the paired calls) and mean P(tumor) (Wilcoxon
    signed-rank over slides).
Holm correction across all endpoint x encoder comparisons.
Output: tables/normalization_tests_20x.csv
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
EVALD = cfg.EVAL_DIR
ENC = {'resnet': 'ResNet50', 'uni': 'UNI', 'conch': 'CONCH'}
AUC_TASKS = {'msi_status': 'MSI (IHC-trained)', 'msi_status_genetic': 'MSI (genetic-trained)',
             'msi_status_consensus': 'MSI (consensus-trained)', 'cptac': 'CPTAC subtype'}
COND = {'raw': 'scale20x_full', 'stainnet': 'scale20xsn_full'}
B = 2000
rng = np.random.default_rng(20260925)


def folds(task, enc, cond):
    fs = sorted(glob.glob(f'{EVALD}/EVAL_cancerscout_colon-cptac_{task}_v3_{enc}_{COND[cond]}/fold_*.csv'))
    assert len(fs) == 10, (task, enc, cond)
    return [pd.read_csv(f).drop_duplicates('slide_id').set_index('slide_id') for f in fs]


def pc(df):
    return sorted([c for c in df.columns if c.startswith('p_')], key=lambda c: int(c[2:]))


def auc(y, P):
    if P.shape[1] == 2:
        return roc_auc_score(y, P[:, 1])
    return float(np.mean([roc_auc_score((y == c).astype(int), P[:, c]) for c in range(P.shape[1])
                          if (y == c).any()]))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(o):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


rows = []
for task, tname in AUC_TASKS.items():
    for enc, ename in ENC.items():
        fr, fs = folds(task, enc, 'raw'), folds(task, enc, 'stainnet')
        cols = pc(fr[0])
        cr = pd.concat(fr).groupby(level=0)[cols + ['Y']].mean()
        cs = pd.concat(fs).groupby(level=0)[cols].mean().loc[cr.index]
        y = cr.Y.round().astype(int).values
        Pr, Ps = cr[cols].values, cs[cols].values
        a_r, a_s = auc(y, Pr), auc(y, Ps)
        case = cr.index.str.split('-').str[0].values
        uc = np.unique(case)
        idx = {c: np.where(case == c)[0] for c in uc}
        diffs = []
        for _ in range(B):
            take = np.concatenate([idx[c] for c in rng.choice(uc, len(uc), replace=True)])
            if len(np.unique(y[take])) < len(np.unique(y)):
                continue
            diffs.append(auc(y[take], Ps[take]) - auc(y[take], Pr[take]))
        diffs = np.array(diffs)
        p_boot = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        per_r = [auc(f.Y.round().astype(int).values, f[cols].values) for f in fr]
        per_s = [auc(f.Y.round().astype(int).values, f[cols].values) for f in fs]
        rows.append(dict(endpoint=tname, encoder=ename, n_slides=len(y), n_cases=len(uc),
                         metric='consensus AUC', raw=a_r, stainnet=a_s, delta=a_s - a_r,
                         ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5),
                         p=p_boot, test='case-level paired bootstrap',
                         fold_mean_raw=np.mean(per_r), fold_mean_stainnet=np.mean(per_s),
                         p_wilcoxon_folds=wilcoxon(per_s, per_r).pvalue))

for enc, ename in ENC.items():
    fr, fs = folds('tumor_vs_normal', enc, 'raw'), folds('tumor_vs_normal', enc, 'stainnet')
    pr = pd.concat(fr).groupby(level=0).p_0.mean()
    ps = pd.concat(fs).groupby(level=0).p_0.mean().loc[pr.index]
    cr, cs = pr > 0.5, ps > 0.5
    b, c = int((~cr & cs).sum()), int((cr & ~cs).sum())
    p_mc = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    case = pr.index.str.split('-').str[0].values
    rows.append(dict(endpoint='Tumor detection', encoder=ename, n_slides=len(pr),
                     n_cases=len(np.unique(case)), metric='share called tumor (P>0.5)',
                     raw=cr.mean(), stainnet=cs.mean(), delta=cs.mean() - cr.mean(),
                     p=p_mc, test=f'exact McNemar ({b} gained, {c} lost)',
                     fold_mean_raw=pr.mean(), fold_mean_stainnet=ps.mean(),
                     p_wilcoxon_folds=wilcoxon(ps, pr).pvalue))

d = pd.DataFrame(rows)
d['p_holm'] = holm(d.p)
d.to_csv(f'{cfg.TABLES}/normalization_tests_20x.csv', index=False)
pd.set_option('display.width', 250)
print(d.drop(columns=['test']).round(4).to_string(index=False))
print(d[['endpoint', 'encoder', 'test']].to_string(index=False))
