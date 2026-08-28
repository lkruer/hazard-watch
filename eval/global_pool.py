"""Can pooled multi-region training be proficient EVERYWHERE?

D16 showed a single-region model exported to a new physiographic regime scores
below chance -- most of the target's feature space was outside training
support. The user's goal is maximum proficiency everywhere on Earth, and the
honest route to that is the LHASA-2.0 route: train ONE model on labels pooled
from every regime we have, so nothing on Earth is far outside support, then
test it the only fair way -- leave-one-region-out (LORO). For each region,
train on all the others and score the holdout cold.

Regions (COOLR satellite inventories + the PNW report catalog):

  myanmar, vietnam, laos, philippines, brazil, malawi, mexico   (inventories)
  pnw                                                            (reports)

Per held-out region the table reports:
  local   5-fold spatial CV trained on the region itself  (ceiling)
  loro    pooled model that never saw the region          (the question)
  loro_rank  same, features as within-region percentiles  (context-normalized)
  nasa    NASA global susceptibility class at our points  (current Tier-B floor)
  slope   univariate slope                                (dumb floor)

Decision rule, stated before running: if LORO beats nasa AND slope in most
regions, the global Tier-B floor should become the pooled model. If LORO sits
at/below the heuristics, Tier B stays heuristic and regional models remain the
only trained tier. Either answer is a result.

Stages (resumable):  --stage build   build per-region matrices (slow, network)
                     --stage nasa    sample NASA map at all points (network)
                     --stage loro    run the evaluation (local compute)
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, RAW, ROOT  # noqa: E402
from eval.spatial_cv import spatial_cv  # noqa: E402
from features import sampling, terrain  # noqa: E402
from features.build_dataset import prefetch_dem  # noqa: E402
from models.train import BASE_PARAMS, make_fit_predict  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

REGIONS = {          # country_name value in coolr_events_points.csv
    "myanmar": "Myanmar",
    "vietnam": "Vietnam",
    "laos": "Laos",
    "philippines": "Philippines",
    "brazil": "Brazil",
    "malawi": "Malawi",
    "mexico": "Mexico",
}
N_POS = 3000
BG_RATIO = 3
SEED = 17
FEATS = list(terrain.FEATURES)          # portable set: terrain only, no road
RUNS = ROOT / "models" / "runs"
NASA_SVC = ("https://gis.earthdata.nasa.gov/gis01/rest/services/Landslides/"
            "Global_Landslide_Susceptibility/ImageServer")


def region_csv(name: str) -> Path:
    return PROCESSED / f"region_{name}.csv"


# ------------------------------------------------------------------ build ---

def events_for(country: str) -> list[dict]:
    out = []
    with (RAW / "coolr_events_points.csv").open(encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if (r.get("country_name") or "").strip() != country:
                continue
            try:
                la, lo = float(r["latitude"]), float(r["longitude"])
            except (TypeError, ValueError):
                continue
            if -90 <= la <= 90 and -180 <= lo <= 180 and not (la == 0 and lo == 0):
                out.append({"lat": la, "lon": lo})
    return out


def build_region(name: str, country: str) -> None:
    out = region_csv(name)
    if out.exists():
        print(f"[{name}] cached ({out.name})")
        return
    pos_all = sampling.dedupe_locations(events_for(country))
    if len(pos_all) < 300:
        print(f"[{name}] only {len(pos_all)} sites -- skipping")
        return
    rng = random.Random(SEED)
    pos = (rng.sample(pos_all, N_POS) if len(pos_all) > N_POS else list(pos_all))

    la = np.array([p["lat"] for p in pos_all])
    lo = np.array([p["lon"] for p in pos_all])
    bbox = (float(np.percentile(lo, 0.5)) - 0.15, float(np.percentile(la, 0.5)) - 0.15,
            float(np.percentile(lo, 99.5)) + 0.15, float(np.percentile(la, 99.5)) + 0.15)
    bg = sampling.target_group_background(bbox, pos_all,
                                          int(len(pos) * BG_RATIO * 1.3), seed=SEED)
    print(f"[{name}] {len(pos_all):,} sites -> {len(pos):,} pos, "
          f"{len(bg):,} bg candidates, bbox {tuple(round(b,2) for b in bbox)}")
    prefetch_dem(pos + bg)

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
        print(f"[{name}]   label={label}: kept {kept:,}")
    terrain.close_all()
    df = pd.DataFrame(sampling.spatial_blocks(rows))
    df["region"] = name
    df.to_csv(out, index=False)
    print(f"[{name}] wrote {len(df):,} rows")


def build_all() -> None:
    for name, country in REGIONS.items():
        # myanmar matrix already exists from the transfer test -- reuse it
        if name == "myanmar" and not region_csv(name).exists():
            src = PROCESSED / "myanmar_terrain.csv"
            if src.exists():
                d = pd.read_csv(src)
                d["region"] = "myanmar"
                d.to_csv(region_csv(name), index=False)
                print("[myanmar] reused transfer-test matrix")
                continue
        build_region(name, country)
    # PNW from the susceptibility matrix (reports regime), terrain columns only
    p = region_csv("pnw")
    if not p.exists():
        d = pd.read_csv(PROCESSED / "susceptibility.csv")
        d = d[["lat", "lon", "label", "block_id"] + [f for f in FEATS if f in d.columns]]
        d["region"] = "pnw"
        d.to_csv(p, index=False)
        print(f"[pnw] wrote {len(d):,} rows from susceptibility.csv")


# ------------------------------------------------------------------- nasa ---

def sample_nasa_for(df: pd.DataFrame, batch: int = 400) -> np.ndarray:
    vals = np.full(len(df), np.nan)
    lats, lons = df["lat"].to_numpy(), df["lon"].to_numpy()
    for i in range(0, len(df), batch):
        pts = [[float(lo), float(la)] for la, lo in
               zip(lats[i:i + batch], lons[i:i + batch])]
        r = SESSION.post(NASA_SVC + "/getSamples", data={
            "geometry": json.dumps({"points": pts,
                                    "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryMultipoint",
            "returnFirstValueOnly": "true", "f": "json"}, timeout=120)
        r.raise_for_status()
        for s in r.json().get("samples", []):
            try:
                vals[i + int(s["locationId"])] = float(str(s["value"]).split()[0])
            except (KeyError, ValueError, IndexError):
                continue
    return vals


def nasa_all() -> None:
    for name in list(REGIONS) + ["pnw"]:
        p = region_csv(name)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "nasa_class" in df.columns and df["nasa_class"].notna().any():
            print(f"[{name}] nasa_class cached")
            continue
        print(f"[{name}] sampling NASA map at {len(df):,} points...")
        df["nasa_class"] = sample_nasa_for(df)
        df.to_csv(p, index=False)


# ------------------------------------------------------------------- loro ---

def rank_within(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f: df[f].rank(pct=True) for f in FEATS})


def loro() -> None:
    frames = {}
    for name in list(REGIONS) + ["pnw"]:
        p = region_csv(name)
        if p.exists():
            frames[name] = pd.read_csv(p)
    print(f"regions available: {', '.join(frames)}  "
          f"({sum(len(d) for d in frames.values()):,} rows pooled)")

    results = {}
    for hold in frames:
        te = frames[hold]
        tr = pd.concat([d for n, d in frames.items() if n != hold],
                       ignore_index=True)
        y_te = te["label"].to_numpy()
        base = float(y_te.mean())
        row = {"n": int(len(te)), "base_rate": base,
               "n_train_regions": len(frames) - 1}

        # pooled, raw features
        m = LGBMClassifier(**BASE_PARAMS).fit(
            tr[FEATS].to_numpy("float64"), tr["label"].to_numpy())
        p = m.predict_proba(te[FEATS].to_numpy("float64"))[:, 1]
        row["loro"] = {"roc_auc": float(roc_auc_score(y_te, p)),
                       "pr_auc": float(average_precision_score(y_te, p))}

        # pooled, within-region rank features
        tr_r = pd.concat([rank_within(d) for n, d in frames.items() if n != hold],
                         ignore_index=True)
        m2 = LGBMClassifier(**BASE_PARAMS).fit(
            tr_r.to_numpy("float64"), tr["label"].to_numpy())
        p2 = m2.predict_proba(rank_within(te).to_numpy("float64"))[:, 1]
        row["loro_rank"] = {"roc_auc": float(roc_auc_score(y_te, p2)),
                            "pr_auc": float(average_precision_score(y_te, p2))}

        # local ceiling (5-fold spatial CV on the region itself)
        folds, oof = spatial_cv(te[FEATS].to_numpy("float64"), y_te,
                                te["block_id"].to_numpy(),
                                make_fit_predict(dict(BASE_PARAMS)), 5)
        mm = np.isfinite(oof)
        row["local"] = {"roc_auc": float(roc_auc_score(y_te[mm], oof[mm])),
                        "pr_auc": float(average_precision_score(y_te[mm], oof[mm]))}

        # every region's local model IS a Tier-A candidate -- persist it,
        # calibrated on its own out-of-fold predictions (pnw already has a
        # richer road-aware artifact from models/train.py; don't overwrite it)
        if hold != "pnw":
            import pickle
            from sklearn.isotonic import IsotonicRegression
            final = LGBMClassifier(**BASE_PARAMS).fit(
                te[FEATS].to_numpy("float64"), y_te)
            iso = IsotonicRegression(out_of_bounds="clip").fit(oof[mm], y_te[mm])
            with (ROOT / "models" / "artifacts" /
                  f"susceptibility-{hold}.pkl").open("wb") as fh:
                pickle.dump({"model": final, "calibrator": iso,
                             "features": FEATS, "region": hold,
                             "cv_roc_auc": row["local"]["roc_auc"]}, fh)

        # floors
        row["slope"] = {"roc_auc": float(roc_auc_score(y_te, te["slope_deg"]))}
        if "nasa_class" in te.columns and te["nasa_class"].notna().sum() > len(te) * 0.5:
            nn = te["nasa_class"].notna().to_numpy()
            if 0 < y_te[nn].mean() < 1:
                row["nasa"] = {"roc_auc": float(
                    roc_auc_score(y_te[nn], te.loc[nn, "nasa_class"]))}
        results[hold] = row
        n_r, l_r = row["loro"]["roc_auc"], row["local"]["roc_auc"]
        print(f"  [{hold:<12}] local {l_r:.3f}  loro {n_r:.3f}  "
              f"rank {row['loro_rank']['roc_auc']:.3f}  "
              f"slope {row['slope']['roc_auc']:.3f}  "
              f"nasa {row.get('nasa', {}).get('roc_auc', float('nan')):.3f}")

    wins = sum(1 for r in results.values()
               if r["loro"]["roc_auc"] > max(r["slope"]["roc_auc"],
                                             r.get("nasa", {}).get("roc_auc", 0)))
    verdict = ("pooled model beats both heuristic floors in "
               f"{wins}/{len(results)} held-out regions")
    print(f"\nverdict: {verdict}")
    out = RUNS / "global-loro.json"
    out.write_text(json.dumps({
        "name": "global-loro", "layer": "susceptibility",
        "protocol": ("leave-one-region-out over pooled multi-region training; "
                     "features terrain-only (portable set)"),
        "features": FEATS, "regions": {k: v for k, v in results.items()},
        "wins": wins, "n_regions": len(results), "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "nasa", "loro", "all"], default="all")
    a = ap.parse_args()
    if a.stage in ("build", "all"):
        build_all()
    if a.stage in ("nasa", "all"):
        nasa_all()
    if a.stage in ("loro", "all"):
        loro()
