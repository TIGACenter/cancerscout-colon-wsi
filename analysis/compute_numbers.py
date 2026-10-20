#!/usr/bin/env python3
"""
Recompute every whole-slide-image number quoted in the CancerScout colon manuscript from the
per-fold prediction files, in-domain (held-out eval.csv) and on CPTAC at 40x and matched 20x.

Definitions (they match the figures, which is what the text must describe):
  * per-fold AUC     binary: ROC AUC of p_1; multiclass: macro-averaged one-vs-rest (this is
                     what CLAM writes to summary.csv and what the figures plot)
  * consensus AUC    AUC of the mean per-fold probability (soft vote over the ten models)
  * balanced acc.    per fold at the threshold fitted on that fold's validation split
                     (summary_best_th_cons.csv, test_threshold_best), argmax for multiclass;
                     consensus = majority vote of the ten per-fold calls, ties broken by mean
                     probability (consensus_vote.py; the stored consensus also counted the true
                     label as a vote)
  * duplicated rows  29 dataset rows point several rescans of one glass slide to the same
                     feature file; they are dropped here (one row per slide_id)
Outputs: tables/in_domain_metrics.csv, tables/cptac_metrics.csv, tables/cptac_perfold.csv
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from consensus_vote import majority_vote

warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cswsi_config as cfg  # noqa: E402
EVALD = cfg.EVAL_DIR
OUT = cfg.TABLES
ENC = ['resnet', 'uni', 'conch']

IN_TASKS = ['tumor_vs_normal', 'sex', 'grading', 'cptac', 'msi_status', 'msi_status_genetic',
            'msi_status_consensus', 'braf', 'k_n_ras', 'tp53', 'apc', 'proteomic_classification',
            'p_g_c2c3c5_cin_cases', 'cin_vs_mcml',
            'ln_metastasis', 'distant_metastasis']
FOLDER = {}
CPTAC_TASKS = ['tumor_vs_normal', 'msi_status', 'msi_status_genetic', 'msi_status_consensus',
               'cptac']
CPTAC_COND = {'20x raw': 'scale20x_full', '20x StainNet': 'scale20xsn_full'}


def load_folds(d):
    fs = sorted(glob.glob(f'{d}/fold_*.csv'), key=lambda p: int(p.split('_')[-1][:-4]))
    return [pd.read_csv(f).drop_duplicates('slide_id').reset_index(drop=True) for f in fs]


def pcols(df):
    return sorted([c for c in df.columns if c.startswith('p_')], key=lambda c: int(c[2:]))


def auc(y, P):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    if P.shape[1] == 2:
        return roc_auc_score(y, P[:, 1])
    present = [c for c in range(P.shape[1]) if (y == c).any()]
    return float(np.mean([roc_auc_score((y == c).astype(int), P[:, c]) for c in present]))


def auc_weighted(y, P):
    y = np.asarray(y).astype(int)
    if P.shape[1] == 2 or len(np.unique(y)) < 2:
        return np.nan
    present = [c for c in range(P.shape[1]) if (y == c).any()]
    w = np.array([(y == c).mean() for c in present])
    return float(np.sum(w * [roc_auc_score((y == c).astype(int), P[:, c]) for c in present]) / w.sum())


def thresholds(task, enc):
    p = f'{EVALD}/EVAL_cancerscout_colon_he_{FOLDER.get(task, task)}_v3_{enc}_s1/summary_best_th_cons.csv'
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p)
    s = s[s.folds.astype(str) != 'consensus']
    return dict(zip(s.folds.astype(int), s.test_threshold_best.astype(float)))


def calls(fold_df, th):
    P = fold_df[pcols(fold_df)].values
    if P.shape[1] == 2:
        return (P[:, 1] > (th if th is not None else 0.5)).astype(int)
    return P.argmax(1)


def summarize(folds, th):
    pc = pcols(folds[0])
    allf = pd.concat([f.assign(fold=i) for i, f in enumerate(folds)])
    cons = allf.groupby('slide_id')[pc + ['Y']].mean()
    y = cons.Y.round().astype(int).values
    per_auc = [auc(f.Y, f[pc].values) for f in folds]
    per_bacc = []
    for i, f in enumerate(folds):
        c = calls(f, th.get(i) if th else None)
        per_bacc.append(balanced_accuracy_score(f.Y.astype(int), c) if f.Y.nunique() > 1 else np.nan)
    maj, ties = majority_vote(folds, [th[i] for i in range(len(folds))] if th else None)
    maj = maj.loc[cons.index]
    return dict(n_slides=len(cons), n_patients=None,
                auc_fold_mean=np.nanmean(per_auc), auc_fold_min=np.nanmin(per_auc),
                auc_fold_max=np.nanmax(per_auc), auc_consensus=auc(y, cons[pc].values),
                auc_consensus_weighted=auc_weighted(y, cons[pc].values),
                bacc_fold_mean=np.nanmean(per_bacc),
                bacc_consensus=balanced_accuracy_score(y, maj) if len(np.unique(y)) > 1 else np.nan,
                vote_ties=ties, per_fold_auc=per_auc, per_fold_bacc=per_bacc,
                called_positive_class0=float((cons['p_0'] > 0.5).mean()),
                # share of slides whose majority vote is class 0 (for tumor detection: tumor)
                called_class0_vote=float((maj == 0).mean()))


def main():
    rows, ipf = [], []
    for t in IN_TASKS:
        for e in ENC:
            d = f'{EVALD}/EVAL_cancerscout_colon_he_{FOLDER.get(t, t)}_v3_{e}_s1'
            if not glob.glob(f'{d}/fold_*.csv'):   # held-out predictions not available (CPTAC-only run)
                continue
            folds = load_folds(d)
            r = summarize(folds, thresholds(t, e))
            r['n_patients'] = pd.concat(folds).slide_id.str.split('_').str[0].nunique()
            for i, (a, b) in enumerate(zip(r['per_fold_auc'], r['per_fold_bacc'])):
                ipf.append(dict(task=t, encoder=e, fold=i, auc=a, bacc=b))
            rows.append(dict(task=t, encoder=e, **{k: v for k, v in r.items() if not k.startswith('per_fold')}))
    ind = pd.DataFrame(rows)
    ind.to_csv(f'{OUT}/in_domain_metrics.csv', index=False)
    pd.DataFrame(ipf).to_csv(f'{OUT}/in_domain_perfold.csv', index=False)

    rows, pf = [], []
    for t in CPTAC_TASKS:
        for e in ENC:
            for cname, suf in CPTAC_COND.items():
                d = f'{EVALD}/EVAL_cancerscout_colon-cptac_{t}_v3_{e}_{suf}'
                if len(glob.glob(f'{d}/fold_*.csv')) != 10:
                    continue
                folds = load_folds(d)
                r = summarize(folds, thresholds(t, e))
                r['n_patients'] = pd.concat(folds).slide_id.str.split('-').str[0].nunique()
                for i, (a, b) in enumerate(zip(r['per_fold_auc'], r['per_fold_bacc'])):
                    pf.append(dict(task=t, encoder=e, condition=cname, fold=i, auc=a, bacc=b))
                rows.append(dict(task=t, encoder=e, condition=cname,
                                 **{k: v for k, v in r.items() if not k.startswith('per_fold')}))
    cp = pd.DataFrame(rows)
    cp.to_csv(f'{OUT}/cptac_metrics.csv', index=False)
    pd.DataFrame(pf).to_csv(f'{OUT}/cptac_perfold.csv', index=False)
    pd.set_option('display.width', 250)
    if len(ind):
        print(ind[['task', 'encoder', 'n_slides', 'auc_fold_mean', 'auc_consensus', 'auc_consensus_weighted',
                   'bacc_fold_mean', 'bacc_consensus', 'vote_ties']].round(3).to_string(index=False))
    if len(cp):
        print(cp[['task', 'encoder', 'condition', 'n_slides', 'n_patients', 'auc_fold_mean', 'auc_consensus',
                  'bacc_consensus', 'called_positive_class0']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
