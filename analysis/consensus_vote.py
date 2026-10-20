"""
Slide-level call of the ten cross-validation models: majority vote of the ten per-model calls
(binary: p_1 above that model's threshold, fitted on its internal test split; multiclass: argmax),
ties broken by the class with the highest mean probability over the ten models. Only the
prediction columns vote; the true label column (Y) of CLAM's fold files is never used.
"""
import numpy as np
import pandas as pd


def per_model_calls(folds, thresholds=None):
    """folds: list of fold DataFrames (slide_id index or column, p_* columns). Returns a slide x
    model DataFrame of integer calls."""
    out = []
    for i, f in enumerate(folds):
        f = f.set_index('slide_id') if 'slide_id' in f.columns else f
        pc = sorted([c for c in f.columns if c.startswith('p_')], key=lambda c: int(c[2:]))
        if len(pc) == 2:
            th = 0.5 if thresholds is None else thresholds[i]
            out.append((f['p_1'] > th).astype(int).rename(i))
        else:
            out.append(pd.Series(f[pc].values.argmax(1), index=f.index, name=i))
    return pd.concat(out, axis=1)


def majority_vote(folds, thresholds=None):
    calls = per_model_calls(folds, thresholds)
    fs = [f.set_index('slide_id') if 'slide_id' in f.columns else f for f in folds]
    pc = sorted([c for c in fs[0].columns if c.startswith('p_')], key=lambda c: int(c[2:]))
    mean_p = pd.concat(fs).groupby(level=0)[pc].mean().loc[calls.index]
    n_cls = len(pc)
    counts = np.stack([(calls.values == k).sum(1) for k in range(n_cls)], 1)
    best = counts.max(1, keepdims=True)
    tied = counts == best
    if n_cls == 2 and thresholds is not None:
        # binary tie: compare the mean probability with the mean threshold
        score = np.stack([np.zeros(len(calls)), mean_p['p_1'].values - np.mean(thresholds)], 1)
    else:
        score = mean_p.values
    score = np.where(tied, score, -np.inf)
    return pd.Series(score.argmax(1), index=calls.index), int((tied.sum(1) > 1).sum())
