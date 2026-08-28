"""Benchmark our susceptibility layer against NASA's published global map.

Stanley & Kirschbaum's Global Landslide Susceptibility map (heuristic fuzzy
combination of slope, geology, faults, forest loss, roads; ~1 km pixels;
classes 1-5) is served live from the same gis01 host as COOLR. Sampling it at
exactly our training points gives an external benchmark nobody here tuned:

  - Both predictors are scored on the SAME labels (our positives + target-group
    background), our model via out-of-fold spatial CV so it never saw the row
    it predicts. PR-AUC of each on identical rows is a fair head-to-head.
  - Spearman rank correlation and the class-wise mean of our score measure
    whether the two maps agree about which terrain is dangerous, independent
    of the labels.

Fairness caveats, both directions: their map is global and coarse (~1 km vs
our 30 m terrain), which handicaps them locally; our labels share the
reporting process our model was fit to (target-group sampling reduces but
does not erase this), which handicaps them differently. Read the comparison
as a sanity check, not a victory lap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, ROOT  # noqa: E402
from eval.spatial_cv import spatial_cv  # noqa: E402
from models.train import BASE_PARAMS, make_fit_predict  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

SERVICE = ("https://gis.earthdata.nasa.gov/gis01/rest/services/Landslides/"
           "Global_Landslide_Susceptibility/ImageServer")
BATCH = 400
RUNS = ROOT / "models" / "runs"


def sample_nasa(lats, lons) -> np.ndarray:
    out = np.full(len(lats), np.nan)
    for i in range(0, len(lats), BATCH):
        pts = [[float(lo), float(la)] for la, lo in
               zip(lats[i:i + BATCH], lons[i:i + BATCH])]
        r = SESSION.post(SERVICE + "/getSamples", data={
            "geometry": json.dumps({"points": pts,
                                    "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryMultipoint",
            "returnFirstValueOnly": "true",
            "f": "json"}, timeout=120)
        r.raise_for_status()
        for s in r.json().get("samples", []):
            try:
                out[i + int(s["locationId"])] = float(str(s["value"]).split()[0])
            except (KeyError, ValueError, IndexError):
                continue
        print(f"  sampled {min(i + BATCH, len(lats)):,}/{len(lats):,}",
              end="\r", flush=True)
    print()
    return out


def main():
    df = pd.read_csv(PROCESSED / "susceptibility.csv")
    feats = [c for c in df.columns if c not in ("lat", "lon", "label", "block_id")
             and df[c].nunique(dropna=True) > 1]
    y = df["label"].to_numpy()

    # tuned params if a tuned run exists, else baseline
    params = dict(BASE_PARAMS)
    tuned = RUNS / "susceptibility-tuned.json"
    if tuned.exists():
        params.update({k: v for k, v in
                       json.loads(tuned.read_text(encoding="utf-8"))["params"].items()})

    print("computing our out-of-fold predictions (spatial CV)...")
    _, oof = spatial_cv(df[feats].to_numpy("float64"), y,
                        df["block_id"].to_numpy(), make_fit_predict(params), 5)

    print("sampling NASA global susceptibility at the same points...")
    nasa = sample_nasa(df["lat"].to_numpy(), df["lon"].to_numpy())

    m = np.isfinite(oof) & np.isfinite(nasa)
    yv, ov, nv = y[m], oof[m], nasa[m]
    base = float(yv.mean())
    print(f"\ncomparable rows: {m.sum():,}   base rate {base:.4f}")

    res = {
        "n": int(m.sum()), "base_rate": base,
        "ours": {"pr_auc": float(average_precision_score(yv, ov)),
                 "roc_auc": float(roc_auc_score(yv, ov))},
        "nasa_global": {"pr_auc": float(average_precision_score(yv, nv)),
                        "roc_auc": float(roc_auc_score(yv, nv))},
        "spearman_ours_vs_nasa": float(spearmanr(ov, nv).statistic),
        "mean_our_score_by_nasa_class": {
            str(int(k)): float(ov[nv == k].mean())
            for k in sorted(np.unique(nv)) if (nv == k).sum() >= 20},
        "nasa_class_counts": {
            str(int(k)): int((nv == k).sum()) for k in sorted(np.unique(nv))},
    }
    print(f"  ours         PR-AUC {res['ours']['pr_auc']:.4f}  "
          f"ROC {res['ours']['roc_auc']:.4f}  (lift {res['ours']['pr_auc']/base:.2f}x)")
    print(f"  NASA global  PR-AUC {res['nasa_global']['pr_auc']:.4f}  "
          f"ROC {res['nasa_global']['roc_auc']:.4f}  "
          f"(lift {res['nasa_global']['pr_auc']/base:.2f}x)")
    print(f"  spearman(our score, NASA class) = {res['spearman_ours_vs_nasa']:+.3f}")
    print("  mean our-score by NASA class:",
          {k: round(v, 3) for k, v in res["mean_our_score_by_nasa_class"].items()})

    out = RUNS / "external-susceptibility.json"
    out.write_text(json.dumps({"name": "external-susceptibility",
                               "layer": "susceptibility", **res},
                              indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
