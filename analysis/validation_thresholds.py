#!/usr/bin/env python3
"""
Per-fold decision thresholds for the binary endpoints, fitted without the held-out set.

For fold i, the threshold maximizing Youden's J (tpr - fpr) on that fold's internal test split of
the training set (results/<experiment>_s1/split_i_results.pkl, written by CLAM at the end of
training) is stored as test_threshold_best in
eval_results/EVAL_<held-out run>/summary_best_th_cons.csv, the file compute_numbers.py reads.
Multiclass endpoints are called by argmax and need no threshold.

The Zenodo model release ships these thresholds as models/clam/thresholds_<encoder>.csv (computed by
this script from the training runs) instead of the split_<fold>_results.pkl files, which list
discovery-cohort slides; when the pkl files are absent, the shipped thresholds are used.
"""
import glob
import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402

eps = pd.read_csv(os.path.join(cfg.REPO, 'configs', 'endpoints.tsv'), sep='\t')
for _, e in eps[eps.multiclass == 0].iterrows():
    for enc in ('resnet', 'uni', 'conch'):
        res = f'{cfg.RESULTS}/cancerscout_colon_he_{e.dataset}_v3_{enc}_s1'
        out = f'{cfg.EVAL_DIR}/EVAL_cancerscout_colon_he_{e.key}_v3_{enc}_s1'
        if not os.path.isdir(res):
            continue
        os.makedirs(out, exist_ok=True)
        if not os.path.exists(f'{res}/split_0_results.pkl'):
            shipped = [pd.read_csv(f) for f in glob.glob(f'{cfg.RESULTS}/thresholds_*.csv')]
            t = pd.concat(shipped) if shipped else pd.DataFrame(columns=['dataset', 'encoder'])
            t = t[(t.dataset == e.dataset) & (t.encoder == enc)].sort_values('fold')
            if len(t) != 10:
                sys.exit(f'no thresholds for {e.dataset} {enc}: neither split pkl files nor thresholds_{enc}.csv')
            t.rename(columns={'fold': 'folds', 'threshold': 'test_threshold_best'})[['folds', 'test_threshold_best']] \
                .to_csv(f'{out}/summary_best_th_cons.csv', index=False)
            print('thresholds (shipped)', e.key, enc)
            continue
        rows = []
        for fold in range(10):
            d = pickle.load(open(f'{res}/split_{fold}_results.pkl', 'rb'))
            y = [float(v['label']) for v in d.values()]
            p = [v['prob'][0][1] for v in d.values()]
            fpr, tpr, th = roc_curve(y, p)
            rows.append({'folds': fold, 'test_threshold_best': th[(tpr - fpr).argmax()]})
        pd.DataFrame(rows).to_csv(f'{out}/summary_best_th_cons.csv', index=False)
        print('thresholds', e.key, enc)
