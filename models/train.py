"""Train the landslide susceptibility and trigger layers.

Gradient boosted trees (LightGBM), per the brief -- a few dozen tabular
features, not raw imagery, is exactly where tree ensembles beat neural nets
while training in seconds and handing back readable feature importance.

Layer separation is deliberate and enforced here:

  susceptibility  terrain features only.  "Where can this happen at all."
  trigger         weather features only.  "Given a susceptible place, is today
                  unusual for this place."

The trigger model is NOT given terrain. Its controls are the same location on
other dates, so terrain is constant within a stratum and cannot help
discriminate -- but pooled across strata the model would happily learn "steep
places have more events", which is the susceptibility signal leaking into the
trigger layer. Keeping the feature sets disjoint keeps the two-layer design
meaningful and keeps the combined score interpretable.

Probabilities are calibrated with isotonic regression fitted on out-of-fold
predictions (never on training-fold predictions, which would be optimistic).

Usage:
    python models/train.py --layer susceptibility
    python models/train.py --layer trigger --tune 40
    python models/train.py --layer all --tune 40
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, ROOT  # noqa: E402
from eval.spatial_cv import (expected_calibration_error, spatial_cv,  # noqa: E402
                             summarise)
from features import terrain  # noqa: E402
from pipelines import openmeteo  # noqa: E402

RUNS = ROOT / "models" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
ARTIFACTS = ROOT / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

LAYERS = {
    # road_dist_m (OSM arterials) rides along when present: reported positives
    # sit a median 20 m from a road vs 887 m for background, and making that
    # accessibility artifact explicit stops elevation from proxying it (D17)
    "susceptibility": {"csv": "susceptibility.csv", "features": terrain.FEATURES,
                       "optional": ["road_dist_m"], "group": "block_id"},
    # grouped on the weather cell, not the 0.25deg block: every row in a cell
    # shares one rainfall series, so a finer group would leak across folds
    "trigger": {"csv": "trigger.csv", "features": list(openmeteo.FEATURES),
                "group": "wx_cell"},
}

BASE_PARAMS = dict(
    objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, n_jobs=-1, verbose=-1,
)

SEARCH = {
    "n_estimators": [200, 400, 700, 1000],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [10, 20, 40, 80],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
    "max_depth": [-1, 4, 6, 10],
}


def load(layer: str):
    cfg = LAYERS[layer]
    p = PROCESSED / cfg["csv"]
    if not p.exists():
        raise SystemExit(f"missing {p}. Run features/build_dataset.py --stage {layer} first.")
    df = pd.read_csv(p)
    feats = [f for f in list(cfg["features"]) + list(cfg.get("optional", []))
             if f in df.columns]
    # drop all-NaN and zero-variance columns, they only add noise
    keep = [f for f in feats if df[f].notna().any() and df[f].nunique(dropna=True) > 1]
    dropped = sorted(set(feats) - set(keep))
    if dropped:
        print(f"  dropping constant/empty features: {', '.join(dropped)}")
    return df, keep


def make_fit_predict(params: dict):
    def fit_predict(Xtr, ytr, Xte):
        m = LGBMClassifier(**params)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    return fit_predict


def cv_score(df, feats, params, n_splits=5, group="block_id"):
    X = df[feats].to_numpy("float64")
    y = df["label"].to_numpy()
    g = df[group].to_numpy()
    folds, oof = spatial_cv(X, y, g, make_fit_predict(params), n_splits=n_splits)
    return folds, oof, summarise(folds, oof, y)


def tune(df, feats, n_iter: int, n_splits: int, group: str = "block_id", seed: int = 17):
    """Random search scored by spatial-CV PR-AUC.

    Random search, not grid: with 8 hyperparameters a grid is astronomically
    large and spends its budget on dimensions that do not matter.
    """
    rng = random.Random(seed)
    best = None
    print(f"  random search: {n_iter} configs x {n_splits} spatial folds")
    for i in range(1, n_iter + 1):
        p = dict(BASE_PARAMS)
        p.update({k: rng.choice(v) for k, v in SEARCH.items()})
        try:
            _, _, s = cv_score(df, feats, p, n_splits, group)
        except Exception as e:                                  # noqa: BLE001
            print(f"    [{i:>3}] failed: {e}")
            continue
        score = s["pr_auc_mean"] or 0.0
        flag = ""
        if best is None or score > best[0]:
            best, flag = (score, p), "  <-- best"
        print(f"    [{i:>3}] PR-AUC {score:.4f} (sd {s['pr_auc_sd']:.4f}){flag}")
    return best[1] if best else dict(BASE_PARAMS)


def train_layer(layer: str, n_tune: int, n_splits: int):
    print(f"\n=== {layer} ===")
    df, feats = load(layer)
    group = LAYERS[layer].get("group", "block_id")
    if group not in df.columns:
        group = "block_id"
    base = df["label"].mean()
    print(f"  rows={len(df):,}  positives={int(df['label'].sum()):,}  "
          f"base rate={base:.4f}  features={len(feats)}  "
          f"CV groups={df[group].nunique():,} ({group})")

    params = dict(BASE_PARAMS)
    tuned = False
    if n_tune > 0:
        params = tune(df, feats, n_tune, n_splits, group)
        tuned = True

    folds, oof, summary = cv_score(df, feats, params, n_splits, group)
    print(f"  PR-AUC {summary['pr_auc_mean']:.4f} +/- {summary['pr_auc_sd']:.4f}"
          f"   (base {base:.4f}, lift {summary['lift_mean']:.2f}x)")
    print(f"  ROC-AUC {summary['roc_auc_mean']:.4f}   Brier {summary['brier_mean']:.4f}"
          f"   ECE {summary['ece_mean']:.4f}")

    # isotonic calibration fitted on out-of-fold predictions only
    y = df["label"].to_numpy()
    m = np.isfinite(oof)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[m], y[m])
    ece_before = expected_calibration_error(y[m], oof[m])
    ece_after = expected_calibration_error(y[m], iso.predict(oof[m]))
    print(f"  calibration ECE {ece_before:.4f} -> {ece_after:.4f} (isotonic on OOF)")

    # final model on all rows, for scoring
    final = LGBMClassifier(**params).fit(df[feats].to_numpy("float64"), y)
    # gain, not the default split count -- split counts just reward whichever
    # feature is most finely divisible and say little about contribution.
    gains = final.booster_.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    imp = sorted(({"feature": f, "gain": float(g), "gain_pct": 100.0 * g / total}
                  for f, g in zip(feats, gains)),
                 key=lambda d: -d["gain"])
    print("  top features by gain: "
          + ", ".join(f"{d['feature']} {d['gain_pct']:.1f}%" for d in imp[:6]))

    run = {
        "name": f"{layer}{'-tuned' if tuned else '-baseline'}",
        "layer": layer,
        "model": "LightGBM",
        "status": "complete",
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "n_pos": int(df["label"].sum()),
        "base_rate": float(base),
        "n_blocks": int(df[group].nunique()),
        "cv": {"scheme": f"GroupKFold on {group}", "n_splits": n_splits},
        "tuned": tuned,
        "n_tune_iter": n_tune,
        "params": {k: (v if not isinstance(v, np.generic) else v.item())
                   for k, v in params.items()},
        "features": feats,
        "feature_importance": imp,
        "folds": [f for f in folds],
        "summary": summary,
        "calibration": {"ece_before": ece_before, "ece_after": ece_after,
                        "method": "isotonic on out-of-fold predictions"},
    }
    out = RUNS / f"{run['name']}.json"
    out.write_text(json.dumps(run, indent=2, default=float), encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}")

    import pickle
    with (ARTIFACTS / f"{layer}.pkl").open("wb") as fh:
        pickle.dump({"model": final, "calibrator": iso, "features": feats}, fh)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["susceptibility", "trigger", "all"], default="all")
    ap.add_argument("--tune", type=int, default=0, help="random-search iterations (0 = baseline only)")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    layers = list(LAYERS) if a.layer == "all" else [a.layer]
    for L in layers:
        try:
            train_layer(L, a.tune, a.folds)
        except SystemExit as e:
            print(f"  skipped: {e}")
