#!/usr/bin/env python3
"""
CPTAC at matched 20x, raw vs StainNet: balanced accuracy of the Fig. 5F endpoints, and whether
normalization moves each model's best CPTAC threshold closer to the threshold it brought from
CancerScout.

Per model (fold) and binary endpoint:
  th_cs    the model's own threshold, fitted on CancerScout (Youden's J on its internal test split;
           summary_best_th_cons.csv), used unchanged on CPTAC
  th_opt   the threshold that would be best on CPTAC for that model (Youden's J on the CPTAC slides;
           an oracle, not something a user could know)
  shift    logit(th_opt) - logit(th_cs): 0 = the CancerScout threshold is already optimal on CPTAC
  bacc_cs  balanced accuracy at th_cs (what the paper reports); bacc_opt at th_opt;
           lost = bacc_opt - bacc_cs, the balanced accuracy lost by transferring the threshold
CPTAC subtype is called by argmax (no threshold). Output: tables/cptac_threshold_migration_20x.csv
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_curve

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
EVALD = cfg.EVAL_DIR
ENC = ['resnet', 'uni', 'conch']
COND = {'raw': 'scale20x_full', 'StainNet': 'scale20xsn_full'}
BINARY = ['msi_status', 'msi_status_genetic', 'msi_status_consensus']


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


rows = []
for t in BINARY + ['cptac']:
    for e in ENC:
        th = None
        if t in BINARY:
            s = pd.read_csv(f'{EVALD}/EVAL_cancerscout_colon_he_{t}_v3_{e}_s1/summary_best_th_cons.csv')
            s = s[s.folds.astype(str) != 'consensus']
            th = dict(zip(s.folds.astype(int), s.test_threshold_best.astype(float)))
        for cname, suf in COND.items():
            for i in range(10):
                f = pd.read_csv(f'{EVALD}/EVAL_cancerscout_colon-cptac_{t}_v3_{e}_{suf}/fold_{i}.csv').drop_duplicates('slide_id')
                y = f.Y.astype(int).values
                if t == 'cptac':
                    pc = [c for c in f.columns if c.startswith('p_')]
                    rows.append(dict(task=t, encoder=e, condition=cname, fold=i,
                                     bacc_cs=balanced_accuracy_score(y, f[pc].values.argmax(1))))
                    continue
                p = f.p_1.values
                fpr, tpr, ths = roc_curve(y, p)
                k = np.argmax(tpr - fpr)
                th_opt = min(ths[k], 1.0)
                rows.append(dict(task=t, encoder=e, condition=cname, fold=i, th_cs=th[i], th_opt=th_opt,
                                 shift=logit(th_opt) - logit(th[i]),
                                 bacc_cs=balanced_accuracy_score(y, (p > th[i]).astype(int)),
                                 bacc_opt=balanced_accuracy_score(y, (p >= th_opt).astype(int)),
                                 sens_cs=((p > th[i]) & (y == 1)).sum() / (y == 1).sum(),
                                 spec_cs=((p <= th[i]) & (y == 0)).sum() / (y == 0).sum()))
d = pd.DataFrame(rows)
d['lost'] = d.bacc_opt - d.bacc_cs
d.to_csv(f'{cfg.TABLES}/cptac_threshold_migration_20x.csv', index=False)
pd.set_option('display.width', 220)
b = d[d.task.isin(BINARY)]
summ = b.groupby(['task', 'encoder', 'condition']).agg(
    median_shift=('shift', 'median'), median_abs_shift=('shift', lambda v: np.median(np.abs(v))),
    bacc_cs=('bacc_cs', 'mean'), bacc_opt=('bacc_opt', 'mean'), lost=('lost', 'mean'),
    sens=('sens_cs', 'mean'), spec=('spec_cs', 'mean')).round(3)
print(summ.to_string())
print(d[d.task == 'cptac'].groupby(['encoder', 'condition']).bacc_cs.mean().round(3).to_string())
