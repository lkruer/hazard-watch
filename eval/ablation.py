"""Feature ablation: how much of the score is real, and how much is bias?

Motivation, from the susceptibility data itself:

    elevation band   positives   background   enrichment
    0-100 m             57.5%        11.3%        5.1x
    800-1600 m           3.8%        35.3%        0.11x

Positives sit far LOWER than background (mean 182 m vs 756 m, standardised
difference -1.27). That is not landslide physics -- steep failure-prone ground
is more common higher up, not at sea level. It is the reporting process:
COOLR is media/report-derived, so events are recorded where roads, towns and
people are, and in this region those sit in valley bottoms and along the coast.

A model handed `elev_m` can therefore score well by learning *where people are*.
This module measures how much of the headline PR-AUC survives when that
shortcut is removed, so the number we publish is the defensible one.

Run:  python eval/ablation.py --layer susceptibility
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, ROOT  # noqa: E402
from eval.spatial_cv import spatial_cv, summarise  # noqa: E402
from models.train import BASE_PARAMS, LAYERS, make_fit_predict  # noqa: E402

SETS = {
    "susceptibility": {
        "all": None,                                    # every feature (incl. road)
        "no_road": ["road_dist_m"],                     # how much does road explain
        "no_elevation": ["elev_m"],
        "no_elevation_no_tpi": ["elev_m", "tpi"],       # keep road: does it substitute?
        "shape_plus_road": ["elev_m", "tpi", "aspect_sin", "aspect_cos"],
        "shape_only": ["elev_m", "tpi", "aspect_sin", "aspect_cos", "road_dist_m"],
    },
    "trigger": {
        "all": None,
        "no_absolute_mm": [c for c in () ],             # filled at runtime
        "percentiles_only": None,                       # filled at runtime
    },
}


def resolve_sets(layer: str, feats: list[str]) -> dict[str, list[str]]:
    if layer == "trigger":
        absolute = [f for f in feats if f.startswith("precip_") and f.endswith("d")]
        pctl = [f for f in feats if "pctl" in f]
        return {
            "all": feats,
            "percentiles_only": pctl,
            "absolute_mm_only": absolute,
        }
    out = {}
    for name, drop in SETS[layer].items():
        out[name] = feats if not drop else [f for f in feats if f not in drop]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=list(LAYERS), default="susceptibility")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    csv = PROCESSED / LAYERS[a.layer]["csv"]
    if not csv.exists():
        raise SystemExit(f"missing {csv}")
    df = pd.read_csv(csv)
    feats = [f for f in (list(LAYERS[a.layer]["features"])
                         + list(LAYERS[a.layer].get("optional", [])))
             if f in df.columns and df[f].notna().any() and df[f].nunique(dropna=True) > 1]

    y = df["label"].to_numpy()
    # Same CV grouping as models/train.py, or the ablation is not comparable to
    # the run it is meant to explain. The trigger layer groups on the weather
    # cell (every row in a cell shares one rainfall series); a finer group leaks.
    group = LAYERS[a.layer].get("group", "block_id")
    if group not in df.columns:
        group = "block_id"
    g = df[group].to_numpy()
    base = float(y.mean())
    print(f"{a.layer}: {len(df):,} rows, base rate {base:.4f}, "
          f"{df[group].nunique():,} CV groups ({group}), {a.folds} folds\n")

    rows = []
    print(f"{'feature set':<22}{'n':>4}{'PR-AUC':>9}{'sd':>8}{'lift':>7}{'ROC':>8}{'vs all':>9}")
    print("-" * 67)
    ref = None
    for name, fs in resolve_sets(a.layer, feats).items():
        if not fs:
            continue
        X = df[fs].to_numpy("float64")
        folds, oof = spatial_cv(X, y, g, make_fit_predict(dict(BASE_PARAMS)),
                                n_splits=a.folds)
        s = summarise(folds, oof, y)
        pr = s["pr_auc_mean"]
        if ref is None:
            ref = pr
        delta = "" if name == "all" else f"{(pr - ref) / ref * 100:+.1f}%"
        print(f"{name:<22}{len(fs):>4}{pr:>9.4f}{s['pr_auc_sd']:>8.4f}"
              f"{pr / base:>6.2f}x{s['roc_auc_mean']:>8.4f}{delta:>9}")
        rows.append({"set": name, "n_features": len(fs), "features": fs,
                     "pr_auc": pr, "pr_auc_sd": s["pr_auc_sd"],
                     "lift": pr / base, "roc_auc": s["roc_auc_mean"]})

    out = ROOT / "models" / "runs" / f"ablation-{a.layer}.json"
    out.write_text(json.dumps({"layer": a.layer, "base_rate": base,
                               "n_rows": int(len(df)), "results": rows},
                              indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
