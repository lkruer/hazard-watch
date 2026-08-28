"""Spatial block cross-validation and honest rare-event metrics.

Shared by every hazard. Two rules drive the whole module:

1. Hold out whole spatial BLOCKS, never random rows. Two landslide points 300 m
   apart have near-identical slope, rainfall and outcome; a random split puts
   one in train and one in test and the model scores brilliantly on information
   it was handed. GroupKFold on a block id, never KFold.

2. PR-AUC is the headline, not accuracy. At a 1-in-6 base rate (and far worse
   for a true daily grid), "always predict no event" already scores >83%
   accuracy while being useless. PR-AUC is always reported next to the base
   rate it has to beat, because PR-AUC alone is not comparable across datasets
   with different positive fractions -- the lift over base rate is the number
   that means something.

Calibration is reported too, since the product serves a probability, not a
ranking: a 0.7 must actually mean roughly 7-in-10.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold


def reliability(y_true, y_prob, n_bins=10):
    """Equal-width reliability curve: predicted vs observed frequency."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not m.any():
            continue
        out.append({"bin_lo": float(lo), "bin_hi": float(hi),
                    "n": int(m.sum()),
                    "mean_pred": float(y_prob[m].mean()),
                    "obs_rate": float(y_true[m].mean())})
    return out


def expected_calibration_error(y_true, y_prob, n_bins=10):
    rows = reliability(y_true, y_prob, n_bins)
    n = len(y_true)
    return float(sum(r["n"] / n * abs(r["mean_pred"] - r["obs_rate"]) for r in rows))


def threshold_for_recall(y_true, y_prob, target_recall=0.80):
    """Lowest-alarm threshold that still catches `target_recall` of events.

    The brief tunes for recall over precision: missing a real landslide is
    worse than a false alarm. We report what that costs in alarm rate so the
    tradeoff is explicit rather than implied.
    """
    order = np.argsort(-y_prob)
    yt = np.asarray(y_true)[order]
    ps = np.asarray(y_prob)[order]
    tp = np.cumsum(yt)
    total_pos = tp[-1] if len(tp) else 0
    if total_pos == 0:
        return None
    recall = tp / total_pos
    idx = np.searchsorted(recall, target_recall, side="left")
    idx = min(idx, len(ps) - 1)
    k = idx + 1
    return {
        "target_recall": target_recall,
        "threshold": float(ps[idx]),
        "achieved_recall": float(recall[idx]),
        "precision": float(tp[idx] / k),
        "alarm_rate": float(k / len(ps)),
    }


def fold_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    base = float(y_true.mean())
    pr = float(average_precision_score(y_true, y_prob))
    return {
        "pr_auc": pr,
        "base_rate": base,
        "lift": pr / base if base > 0 else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if 0 < base < 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def spatial_cv(X, y, groups, fit_predict, n_splits=5):
    """Run GroupKFold over spatial blocks.

    `fit_predict(X_tr, y_tr, X_te) -> probabilities` keeps this harness model
    agnostic, so the same code evaluates LightGBM here and anything else later.
    Returns (per-fold metrics, out-of-fold probabilities).
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    groups = np.asarray(groups)

    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"only {n_groups} spatial blocks for {n_splits} folds")

    oof = np.full(len(y), np.nan)
    folds = []
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        if y[te].sum() == 0 or y[tr].sum() == 0:
            folds.append({"fold": k, "skipped": "no positives in split",
                          "n": int(len(te))})
            continue
        p = fit_predict(X[tr], y[tr], X[te])
        oof[te] = p
        m = fold_metrics(y[te], p)
        m["fold"] = k
        m["n_train"] = int(len(tr))
        m["n_blocks_test"] = int(len(np.unique(groups[te])))
        folds.append(m)
    return folds, oof


def summarise(folds, oof, y):
    """Aggregate fold metrics plus pooled out-of-fold performance."""
    ok = [f for f in folds if "pr_auc" in f]
    m = np.isfinite(oof)
    pooled = fold_metrics(np.asarray(y)[m], oof[m]) if m.any() else {}
    def avg(k):
        v = [f[k] for f in ok if isinstance(f.get(k), (int, float))]
        return float(np.mean(v)) if v else None
    def sd(k):
        v = [f[k] for f in ok if isinstance(f.get(k), (int, float))]
        return float(np.std(v)) if len(v) > 1 else 0.0
    return {
        "n_folds": len(ok),
        "pr_auc_mean": avg("pr_auc"), "pr_auc_sd": sd("pr_auc"),
        "roc_auc_mean": avg("roc_auc"), "brier_mean": avg("brier"),
        "ece_mean": avg("ece"), "lift_mean": avg("lift"),
        "pooled_oof": pooled,
        "reliability": reliability(np.asarray(y)[m], oof[m]) if m.any() else [],
        "recall_threshold": (threshold_for_recall(np.asarray(y)[m], oof[m], 0.80)
                             if m.any() else None),
    }
