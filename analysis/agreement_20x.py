#!/usr/bin/env python3
"""
Does StainNet make the ten models agree more on CPTAC? Per slide: agreement = share of the ten models
whose call equals the majority vote (calls as in the Methods: each model at its CancerScout threshold,
argmax for CPTAC subtype); unanimous = all ten agree. Raw vs StainNet at 20x on the same slides;
difference in mean agreement tested by resampling patients (2,000 bootstrap resamples), Holm over
the 15 endpoint-encoder comparisons. Output: tables/agreement_20x.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from consensus_vote import majority_vote, per_model_calls  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import cswsi_config as cfg  # noqa: E402
EVALD = cfg.EVAL_DIR
ENC = {'resnet': 'ResNet50', 'uni': 'UNI', 'conch': 'CONCH'}
TASKS = {'tumor_vs_normal': 'Tumor detection', 'msi_status': 'MSI (IHC-trained)',
         'msi_status_genetic': 'MSI (genetic-trained)', 'msi_status_consensus': 'MSI (consensus-trained)',
         'cptac': 'CPTAC subtype'}
rng = np.random.default_rng(20260927)


def agreement(t, e, suf):
    th = None
    if t != 'cptac':
        s = pd.read_csv(f'{EVALD}/EVAL_cancerscout_colon_he_{t}_v3_{e}_s1/summary_best_th_cons.csv')
        th = list(s[s.folds.astype(str) != 'consensus'].test_threshold_best.astype(float))
    fs = [pd.read_csv(f).drop_duplicates('slide_id') for f in
          sorted(glob.glob(f'{EVALD}/EVAL_cancerscout_colon-cptac_{t}_v3_{e}_{suf}/fold_*.csv'),
                 key=lambda p: int(p.split('_')[-1][:-4]))]
    calls = per_model_calls(fs, th)
    vote, _ = majority_vote(fs, th)
    return pd.Series((calls.values == vote.loc[calls.index].values[:, None]).mean(1), index=calls.index)


rows = []
for t, tn in TASKS.items():
    for e, en in ENC.items():
        r = agreement(t, e, 'scale20x_full')
        s = agreement(t, e, 'scale20xsn_full').loc[r.index]
        case = r.index.str.split('-').str[0].values
        uc = np.unique(case)
        idx = {c: np.where(case == c)[0] for c in uc}
        dv = (s - r).values
        boots = np.array([dv[np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(2000)])
        rows.append(dict(endpoint=tn, encoder=en, agreement_raw=r.mean(), agreement_stainnet=s.mean(),
                         delta=s.mean() - r.mean(), unanimous_raw=(r == 1).mean(), unanimous_stainnet=(s == 1).mean(),
                         p=min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))))
d = pd.DataFrame(rows)
o, adj, run = np.argsort(d.p.values), np.empty(len(d)), 0
for rank, i in enumerate(o):
    run = max(run, (len(d) - rank) * d.p.values[i])
    adj[i] = min(1, run)
d['p_holm'] = adj
d['significant'] = np.where(d.p_holm < 0.05, 'yes', 'no')
d.to_csv(f'{cfg.TABLES}/agreement_20x.csv', index=False)
pd.set_option('display.width', 200)
print(d.round(3).to_string(index=False))
print(d.groupby('encoder')[['delta']].mean().round(3))
