#!/usr/bin/env python3
"""
CPTAC at 20x, raw vs StainNet, with the Methods' call rule (each model at its CancerScout threshold,
majority vote of the ten models): balanced accuracy (MSI x3, CPTAC subtype) and the share of slides
called tumor. Paired over the same slides; balanced-accuracy differences tested by resampling
patients (2,000 bootstrap resamples), tumor calls by an exact McNemar test; Holm correction over all
comparisons. Output: tables/bacc_tests_20x.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from consensus_vote import majority_vote  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import cswsi_config as cfg  # noqa: E402
EVALD = cfg.EVAL_DIR
ENC = {'resnet': 'ResNet50', 'uni': 'UNI', 'conch': 'CONCH'}
TASKS = {'msi_status': 'MSI (IHC-trained)', 'msi_status_genetic': 'MSI (genetic-trained)',
         'msi_status_consensus': 'MSI (consensus-trained)', 'cptac': 'CPTAC subtype',
         'tumor_vs_normal': 'Tumor detection'}
rng = np.random.default_rng(20260926)


def thresholds(t, e):
    if t == 'cptac':
        return None
    s = pd.read_csv(f'{EVALD}/EVAL_cancerscout_colon_he_{t}_v3_{e}_s1/summary_best_th_cons.csv')
    s = s[s.folds.astype(str) != 'consensus']
    return list(s.test_threshold_best.astype(float))


def vote(t, e, suf):
    fs = [pd.read_csv(f).drop_duplicates('slide_id') for f in
          sorted(glob.glob(f'{EVALD}/EVAL_cancerscout_colon-cptac_{t}_v3_{e}_{suf}/fold_*.csv'),
                 key=lambda p: int(p.split('_')[-1][:-4]))]
    v, _ = majority_vote(fs, thresholds(t, e))
    y = fs[0].set_index('slide_id').Y.astype(int).loc[v.index]
    return v, y


rows = []
for t, tn in TASKS.items():
    for e, en in ENC.items():
        vr, y = vote(t, e, 'scale20x_full')
        vs, _ = vote(t, e, 'scale20xsn_full')
        vs = vs.loc[vr.index]
        if t == 'tumor_vs_normal':
            r, s = (vr == 0), (vs == 0)          # class 0 = tumor
            g, l = int((~r & s).sum()), int((r & ~s).sum())
            rows.append(dict(endpoint=tn, encoder=en, metric='share called tumor', raw=r.mean(), stainnet=s.mean(),
                             delta=s.mean() - r.mean(), p=binomtest(g, g + l, 0.5).pvalue if g + l else 1.0))
            continue
        a_r, a_s = balanced_accuracy_score(y, vr), balanced_accuracy_score(y, vs)
        case = vr.index.str.split('-').str[0].values
        uc = np.unique(case)
        idx = {c: np.where(case == c)[0] for c in uc}
        yv, rv, sv = y.values, vr.values, vs.values
        diffs = []
        for _ in range(2000):
            k = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
            if len(np.unique(yv[k])) < len(np.unique(yv)):
                continue
            diffs.append(balanced_accuracy_score(yv[k], sv[k]) - balanced_accuracy_score(yv[k], rv[k]))
        diffs = np.array(diffs)
        rows.append(dict(endpoint=tn, encoder=en, metric='balanced accuracy', raw=a_r, stainnet=a_s,
                         delta=a_s - a_r, ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5),
                         p=min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))))
d = pd.DataFrame(rows)
o = np.argsort(d.p.values)
adj, run = np.empty(len(d)), 0
for rank, i in enumerate(o):
    run = max(run, (len(d) - rank) * d.p.values[i])
    adj[i] = min(1, run)
d['p_holm'] = adj
d.to_csv(f'{cfg.TABLES}/bacc_tests_20x.csv', index=False)
pd.set_option('display.width', 200)
print(d.round(3).to_string(index=False))
