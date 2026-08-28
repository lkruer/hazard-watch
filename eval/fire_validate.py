"""Validate the fire-danger layer against two continents of public fire records.

Design mirrors the landslide trigger exactly -- case-crossover, so location,
fuel type, and human activity cancel within each stratum and only weather
separates a fire day from its season-matched control days at the same spot:

  US labels      FPA-FOD 6th ed. (Short, USDA RDS-2013-0009.6, public domain):
                 2.3M wildfires 1992-2020. We use discoveries >= 100 acres.
  Canada labels  NFDB point catalog (CWFIS, open): fires >= 100 ha.

The model never sees a label at build time -- KBDI/VPD/percentiles are pure
weather functions -- so the labels only measure how much danger signal the
features carry. Three questions:

  1. US skill: 5-fold CV grouped by weather cell.
  2. Does it TRAVEL? Train US -> test boreal Canada cold (the fire analogue of
     the landslide LORO). Fire-weather physics is more universal than terrain
     regimes, so unlike D16 this transfer is expected to hold; measuring it is
     the point.
  3. Do single indices (KBDI, VPD percentiles) already carry the skill, or
     does the model add value? (D14's lesson, re-asked for fire.)

Stages:  --stage build   extract labels, build case-crossover strata
         --stage eval    fetch weather, features, train/validate, artifacts
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import pickle
import random
import sqlite3
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, RAW, ROOT  # noqa: E402
from eval.spatial_cv import spatial_cv  # noqa: E402
from features import sampling  # noqa: E402
from models.train import BASE_PARAMS, make_fit_predict  # noqa: E402
from pipelines import fireweather as fw  # noqa: E402

FIRE_RAW = RAW / "fire"
SEED = 17
MIN_ACRES_US = 100.0
MIN_HA_CA = 100.0
MAX_CELLS = {"us": 320, "canada": 220}
PER_CELL = 4
CONTROLS = 4
RUNS = ROOT / "models" / "runs"


# ------------------------------------------------------------------ labels ---

def us_fires() -> pd.DataFrame:
    """FPA-FOD discoveries >= MIN_ACRES_US within the weather span."""
    zp = FIRE_RAW / "fpa_fod_sqlite.zip"
    db = FIRE_RAW / "fpa_fod.sqlite"
    if not db.exists():
        with zipfile.ZipFile(zp) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".sqlite"))
            print(f"  extracting {name} ...")
            with z.open(name) as src, db.open("wb") as dst:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    dst.write(b)
    con = sqlite3.connect(str(db))
    cols = {r[1].upper() for r in con.execute("PRAGMA table_info(Fires)")}
    date_expr = ("DISCOVERY_DATE" if "DISCOVERY_DATE" in cols else None)
    q = (f"SELECT LATITUDE lat, LONGITUDE lon, FIRE_SIZE size_acres, "
         f"FIRE_YEAR yr, DISCOVERY_DOY doy, {date_expr} ddate "
         f"FROM Fires WHERE FIRE_SIZE >= {MIN_ACRES_US}")
    df = pd.read_sql_query(q, con)
    con.close()

    def to_iso(row):
        d = row["ddate"]
        if isinstance(d, str) and len(d) >= 10:
            s = d[:10]
            if s[4] == "-":                       # YYYY-MM-DD
                return s
            if s[2] == "/":                       # MM/DD/YYYY
                return f"{s[6:10]}-{s[0:2]}-{s[3:5]}"
        if isinstance(d, (int, float)) and d > 2_400_000:   # julian day
            return (dt.date(1858, 11, 17)
                    + dt.timedelta(days=float(d) - 2_400_000.5)).isoformat()
        try:
            return (dt.date(int(row["yr"]), 1, 1)
                    + dt.timedelta(days=int(row["doy"]) - 1)).isoformat()
        except (TypeError, ValueError):
            return None

    df["date"] = df.apply(to_iso, axis=1)
    df = df.dropna(subset=["date", "lat", "lon"])
    df = df[(df["date"] >= "2005-01-01") & (df["date"] <= fw.END)]
    print(f"  US fires >= {MIN_ACRES_US:.0f} acres in span: {len(df):,}")
    return df[["lat", "lon", "date"]]


def ca_fires() -> pd.DataFrame:
    zp = FIRE_RAW / "nfdb_point_txt.zip"
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
        with z.open(name) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
                             low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    lat = df["LATITUDE"]
    lon = df["LONGITUDE"]
    size = pd.to_numeric(df.get("SIZE_HA"), errors="coerce")
    rep = df.get("REP_DATE").astype(str).str.slice(0, 10)
    ok = rep.str.match(r"\d{4}-\d{2}-\d{2}", na=False)
    out = pd.DataFrame({"lat": lat, "lon": lon, "date": rep, "ha": size})[ok]
    out = out[(out["ha"] >= MIN_HA_CA)
              & (out["date"] >= "2005-01-01") & (out["date"] <= fw.END)]
    out = out.dropna(subset=["lat", "lon"])
    print(f"  Canada fires >= {MIN_HA_CA:.0f} ha in span: {len(out):,}")
    return out[["lat", "lon", "date"]]


def build_strata(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Cell-clustered case-crossover skeleton (no weather yet)."""
    rng = random.Random(SEED)
    by_cell = defaultdict(list)
    for r in df.itertuples(index=False):
        by_cell[fw.cell(r.lat, r.lon)].append((r.lat, r.lon, r.date))
    dense = {c: v for c, v in by_cell.items() if len(v) >= 4}
    cells = sorted(dense)
    rng.shuffle(cells)
    cells = cells[:MAX_CELLS[tag]]
    print(f"  [{tag}] {len(by_cell):,} cells -> {len(dense):,} dense -> "
          f"using {len(cells)}")
    rows = []
    for c in cells:
        fires = dense[c]
        rng.shuffle(fires)
        hist = [d for (_, _, d) in dense[c]]
        for (la, lo, d) in fires[:PER_CELL]:
            sid = f"{tag}_{la:.3f}_{lo:.3f}_{d}"
            rows.append({"lat": la, "lon": lo, "date": d, "label": 1,
                         "stratum": sid, "wx_cell": f"{c[0]}_{c[1]}", "domain": tag})
            for cd in sampling.control_dates(
                    d, hist, n=CONTROLS, seed=SEED, season_window=45,
                    exclusion_days=21, year_min=2006,
                    year_max=int(fw.END[:4])):
                rows.append({"lat": la, "lon": lo, "date": cd, "label": 0,
                             "stratum": sid, "wx_cell": f"{c[0]}_{c[1]}",
                             "domain": tag})
    out = pd.DataFrame(rows)
    print(f"  [{tag}] strata rows: {len(out):,} ({int(out.label.sum()):,} cases)")
    return out


def stage_build() -> None:
    parts = []
    if (FIRE_RAW / "fpa_fod_sqlite.zip").exists():
        parts.append(build_strata(us_fires(), "us"))
    else:
        print("  FPA-FOD zip not present -- skipping US")
    if (FIRE_RAW / "nfdb_point_txt.zip").exists():
        parts.append(build_strata(ca_fires(), "canada"))
    else:
        print("  NFDB zip not present -- skipping Canada")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(PROCESSED / "fire_strata.csv", index=False)
    print(f"wrote {len(df):,} rows -> fire_strata.csv")


# -------------------------------------------------------------------- eval ---

def stage_eval() -> None:
    df = pd.read_csv(PROCESSED / "fire_strata.csv")
    cells = sorted({fw.cell(r.lat, r.lon) for r in df.itertuples(index=False)})
    print(f"weather cells needed: {len(cells)}")
    fw.fetch_cells(cells)

    rows, miss = [], 0
    for r in df.itertuples(index=False):
        f = fw.features_at(r.lat, r.lon, r.date)
        if f is None:
            miss += 1
            continue
        rows.append({**r._asdict(), **f})
    d = pd.DataFrame(rows)
    print(f"rows with weather: {len(d):,} (dropped {miss:,})")

    ok = d.groupby("stratum")["label"].agg(["max", "min"])
    keep = ok[(ok["max"] == 1) & (ok["min"] == 0)].index
    d = d[d["stratum"].isin(keep)].reset_index(drop=True)
    d.to_csv(PROCESSED / "fire_trigger.csv", index=False)
    feats = list(fw.FEATURES)
    res = {}

    def auc_block(sub, p, name):
        y = sub["label"].to_numpy()
        res[name] = {"n": int(len(sub)), "n_cases": int(y.sum()),
                     "roc_auc": float(roc_auc_score(y, p)),
                     "pr_auc": float(average_precision_score(y, p)),
                     "base_rate": float(y.mean())}
        print(f"  {name:<22} ROC {res[name]['roc_auc']:.4f}  "
              f"PR {res[name]['pr_auc']:.4f} (base {res[name]['base_rate']:.3f})")

    for dom in ("us", "canada"):
        sub = d[d.domain == dom]
        if not len(sub):
            continue
        print(f"\n=== {dom} ===  ({sub.wx_cell.nunique()} cells)")
        folds, oof = spatial_cv(sub[feats].to_numpy("float64"),
                                sub["label"].to_numpy(),
                                sub["wx_cell"].to_numpy(),
                                make_fit_predict(dict(BASE_PARAMS)), 5)
        m = np.isfinite(oof)
        auc_block(sub[m], oof[m], f"{dom}_local_cv")
        for idx_name in ("kbdi_pctl_seasonal", "vpd_pctl_seasonal"):
            auc_block(sub, sub[idx_name].fillna(0.5).to_numpy(),
                      f"{dom}_{idx_name}")

    us, ca = d[d.domain == "us"], d[d.domain == "canada"]
    if len(us) and len(ca):
        print("\n=== transfer: train US -> test Canada ===")
        mdl = LGBMClassifier(**BASE_PARAMS).fit(
            us[feats].to_numpy("float64"), us["label"].to_numpy())
        auc_block(ca, mdl.predict_proba(ca[feats].to_numpy("float64"))[:, 1],
                  "us_to_canada_transfer")
        mdl2 = LGBMClassifier(**BASE_PARAMS).fit(
            ca[feats].to_numpy("float64"), ca["label"].to_numpy())
        auc_block(us, mdl2.predict_proba(us[feats].to_numpy("float64"))[:, 1],
                  "canada_to_us_transfer")

    # final artifact: pooled both domains, calibrated on pooled OOF
    folds, oof = spatial_cv(d[feats].to_numpy("float64"), d["label"].to_numpy(),
                            d["wx_cell"].to_numpy(),
                            make_fit_predict(dict(BASE_PARAMS)), 5)
    m = np.isfinite(oof)
    auc_block(d[m], oof[m], "pooled_cv")
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        oof[m], d["label"].to_numpy()[m])
    final = LGBMClassifier(**BASE_PARAMS).fit(
        d[feats].to_numpy("float64"), d["label"].to_numpy())
    gains = final.booster_.feature_importance(importance_type="gain")
    imp = sorted(({"feature": f, "gain_pct": 100.0 * g / max(1e-9, gains.sum())}
                  for f, g in zip(feats, gains)), key=lambda x: -x["gain_pct"])
    print("\ntop features: " + ", ".join(
        f"{i['feature']} {i['gain_pct']:.1f}%" for i in imp[:6]))
    with (ROOT / "models" / "artifacts" / "fire_trigger.pkl").open("wb") as fh:
        pickle.dump({"model": final, "calibrator": iso, "features": feats,
                     "domains": ["us", "canada"]}, fh)

    (RUNS / "fire-trigger.json").write_text(json.dumps({
        "name": "fire-trigger", "layer": "fire", "model": "LightGBM",
        "status": "complete",
        "framing": "fire danger conditions (not ignition), case-crossover",
        "labels": {"us": "FPA-FOD 6th ed >=100 acres (public domain)",
                   "canada": f"NFDB points >={MIN_HA_CA:.0f} ha (open)"},
        "features": feats, "results": res,
        "feature_importance": imp,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"wrote models/runs/fire-trigger.json + artifacts/fire_trigger.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "eval", "all"], default="all")
    a = ap.parse_args()
    if a.stage in ("build", "all"):
        stage_build()
    if a.stage in ("eval", "all"):
        stage_eval()
