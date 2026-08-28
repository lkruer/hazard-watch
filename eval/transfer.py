"""Transfer test: does the PNW-trained terrain model work where labels have
no reporting bias at all?

The Myanmar (Chin/Rakhine) inventory in COOLR is satellite-mapped: analysts
digitised every visible failure in the imagery footprint after major storms.
7,970 events with no roads-and-reporters filter -- the one label source we
have that cannot share the PNW's observation bias. If the terrain model
learned physics, it should rank these slide sites above matched background in
a country it has never seen, on different lithology, in a different climate.

Three predictors on identical Myanmar rows:
  transfer   the PNW-trained terrain-only model, frozen
  local      a LightGBM trained on Myanmar itself (5-fold spatial CV) --
             the ceiling for this feature set here
  slope      univariate slope ranking -- the floor any terrain model must beat

Expectations, stated before running: transfer should land well above chance
and slope, below local. A transfer ROC near 0.5 would mean the PNW model is
regional statistics dressed up as physics; near-local would mean terrain
signal generalises almost fully.
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, ROOT  # noqa: E402
from eval.spatial_cv import spatial_cv  # noqa: E402
from features import sampling, terrain  # noqa: E402
from models.train import BASE_PARAMS, make_fit_predict  # noqa: E402

BBOX = (92.0, 21.75, 94.0, 23.75)          # from pipelines/select_region.py
N_POS = 3000
BG_RATIO = 3
SEED = 17
RUNS = ROOT / "models" / "runs"


def load_inventory():
    import csv
    x0, y0, x1, y1 = BBOX
    out = []
    with (RAW / "coolr_events_points.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                la, lo = float(r["latitude"]), float(r["longitude"])
            except (TypeError, ValueError):
                continue
            if x0 <= lo <= x1 and y0 <= la <= y1:
                out.append({"lat": la, "lon": lo})
    return out


def main():
    pos_all = sampling.dedupe_locations(load_inventory())
    rng = random.Random(SEED)
    pos = rng.sample(pos_all, min(N_POS, len(pos_all)))
    print(f"inventory: {len(pos_all):,} distinct sites, using {len(pos):,}")

    # anchor background on ALL sites so the footprint is fully represented;
    # min-dist exclusion also uses all sites so no 'background' sits on a slide
    bg = sampling.target_group_background(BBOX, pos_all,
                                          int(len(pos) * BG_RATIO * 1.3),
                                          seed=SEED)
    print(f"background candidates: {len(bg):,}")

    from config import PROCESSED
    cache = PROCESSED / "myanmar_terrain.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"terrain matrix from cache: {len(df):,} rows")
    else:
        rows = []
        for label, pts, cap in ((1, pos, len(pos)), (0, bg, len(pos) * BG_RATIO)):
            kept = 0
            for p in pts:
                if kept >= cap:
                    break
                t = terrain.derive(p["lat"], p["lon"])
                if t is None or (t["elev_m"] <= 0.5 and t["roughness_std"] < 0.1):
                    continue
                rows.append({"lat": p["lat"], "lon": p["lon"], "label": label, **t})
                kept += 1
            print(f"  label={label}: kept {kept:,}")
        terrain.close_all()
        df = pd.DataFrame(sampling.spatial_blocks(rows))
        df.to_csv(cache, index=False)
    y = df["label"].to_numpy()
    base = float(y.mean())
    print(f"rows {len(df):,}  base {base:.4f}  blocks {df.block_id.nunique()}")

    with (ROOT / "models" / "artifacts" / "susceptibility-terrain-only.pkl").open("rb") as fh:
        b = pickle.load(fh)
    feats = b["features"]
    X = df[feats].to_numpy("float64")

    p_transfer = b["model"].predict_proba(X)[:, 1]
    res = {"n": int(len(df)), "base_rate": base,
           "transfer": {"roc_auc": float(roc_auc_score(y, p_transfer)),
                        "pr_auc": float(average_precision_score(y, p_transfer))}}

    print("training local Myanmar benchmark (5-fold spatial CV)...")
    folds, oof = spatial_cv(X, y, df["block_id"].to_numpy(),
                            make_fit_predict(dict(BASE_PARAMS)), 5)
    m = np.isfinite(oof)
    res["local"] = {"roc_auc": float(roc_auc_score(y[m], oof[m])),
                    "pr_auc": float(average_precision_score(y[m], oof[m]))}
    res["slope_only"] = {
        "roc_auc": float(roc_auc_score(y, df["slope_deg"])),
        "pr_auc": float(average_precision_score(y, df["slope_deg"]))}

    # Attribution: which PNW-learned relationships transfer and which invert?
    # Retrain PNW models on feature subsets, score Myanmar with each.
    pnw = pd.read_csv(PROCESSED / "susceptibility.csv")
    subsets = {
        "pnw_shape_only": [f for f in feats if f not in
                           ("elev_m", "tpi", "aspect_sin", "aspect_cos")],
        "pnw_no_elev_tpi": [f for f in feats if f not in ("elev_m", "tpi")],
        "pnw_elev_tpi_only": ["elev_m", "tpi"],
    }
    for name, fs in subsets.items():
        mm = LGBMClassifier(**BASE_PARAMS).fit(
            pnw[fs].to_numpy("float64"), pnw["label"].to_numpy())
        pp = mm.predict_proba(df[fs].to_numpy("float64"))[:, 1]
        res[name] = {"features": fs,
                     "roc_auc": float(roc_auc_score(y, pp)),
                     "pr_auc": float(average_precision_score(y, pp))}

    print(f"\n{'predictor':<12}{'ROC-AUC':>9}{'PR-AUC':>9}{'lift':>7}")
    for k in ("transfer", "pnw_no_elev_tpi", "pnw_shape_only",
              "pnw_elev_tpi_only", "local", "slope_only"):
        r = res[k]
        print(f"{k:<12}{r['roc_auc']:>9.4f}{r['pr_auc']:>9.4f}"
              f"{r['pr_auc']/base:>6.2f}x")

    out = RUNS / "transfer-susceptibility.json"
    out.write_text(json.dumps({"name": "transfer-susceptibility",
                               "layer": "susceptibility",
                               "region": "Myanmar (Chin/Rakhine)", "bbox": BBOX,
                               **res}, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
