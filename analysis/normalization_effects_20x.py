#!/usr/bin/env python3
"""
All raw-vs-StainNet comparisons at 20x in one table, with the paper's rules: AUC of the averaged
prediction (normalization_tests.py) and, for calls, the majority vote of the ten models at their
CancerScout thresholds (bacc_tests_20x.py). One Holm correction over all 27 comparisons.
Output: tables/normalization_effects_20x.csv
"""
import os

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
T = cfg.TABLES
auc = pd.read_csv(f'{T}/normalization_tests_20x.csv')
auc = auc[auc.endpoint != 'Tumor detection'].assign(metric='AUC')
calls = pd.read_csv(f'{T}/bacc_tests_20x.csv')
d = pd.concat([auc[['endpoint', 'encoder', 'metric', 'raw', 'stainnet', 'delta', 'p']],
               calls[['endpoint', 'encoder', 'metric', 'raw', 'stainnet', 'delta', 'p']]], ignore_index=True)
o, adj, run = np.argsort(d.p.values), np.empty(len(d)), 0
for rank, i in enumerate(o):
    run = max(run, (len(d) - rank) * d.p.values[i])
    adj[i] = min(1, run)
d['p_holm'] = adj
d['significant'] = np.where(d.p_holm < 0.05, 'yes', 'no')
d.to_csv(f'{T}/normalization_effects_20x.csv', index=False)
pd.set_option('display.width', 200)
print(d.round(3).to_string(index=False))
print('mean change by encoder and metric:')
print(d.groupby(['metric', 'encoder']).delta.mean().round(3).unstack())
