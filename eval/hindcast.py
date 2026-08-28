"""Prospective hindcast: would this warning system have worked, run for real?

The trigger model's PR-AUC was measured on a constructed 1-case-to-4-controls
dataset. Real deployment scores EVERY day, where reported-event days are about
7-in-10,000 -- so that number says how well the model *ranks* days, not what
operating it feels like. This module answers the operational question the way
forecast verification actually does it:

  1. Freeze everything on data through 2015: re-tune on pre-2016 strata only
     (controls dated after 2015 are dropped too), fit, calibrate isotonically
     on pre-2016 out-of-fold predictions, and take the 80%-recall operating
     threshold from that same OOF. Nothing from 2016+ touches the model.
  2. Replay 2016-2024: score every day at every one of the 92 weather cells
     (~300k cell-days) and raise alarms where the frozen threshold is crossed.
  3. Score the alarms against reported events with the metrics operational
     systems use: POD (fraction of event-days alarmed), FAR (fraction of
     alarms with no event), alarm days per cell per year.
  4. Compare against a no-ML baseline -- "alarm when 3-day rainfall crosses a
     seasonal percentile" -- swept over thresholds, so the value added by the
     model over a rule anyone could implement in a spreadsheet is measured,
     not assumed.

Honesty notes baked into the output:
  - Events are REPORTED events. An alarm on a real-but-unreported slide counts
    as false, so FAR here is an upper bound; POD is measured on reported
    events only.
  - The 92 cells are cells where at least one event was ever reported --
    behaviour in never-reporting terrain is not measured.
  - A +/-1 day tolerance is reported alongside strict matching, since report
    dates can lag occurrence by a day.
"""
from __future__ import annotations

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
from eval.spatial_cv import spatial_cv, threshold_for_recall  # noqa: E402
from features.build_dataset import ACC_WEATHER, load_reports, region  # noqa: E402
from models.train import BASE_PARAMS, SEARCH  # noqa: E402
from pipelines import nasapower  # noqa: E402
from pipelines.openmeteo import FEATURES, WINDOWS, CellSeries  # noqa: E402

SPLIT_YEAR = 2015          # train <= this, evaluate strictly after
SEED = 17
RUNS = ROOT / "models" / "runs"


# ---------------------------------------------------- vectorized features ---

def features_all(cs: CellSeries) -> tuple[list[str], np.ndarray]:
    """Whole-series feature matrix, exactly mirroring CellSeries.features().

    Returns (dates, X) where X[i] holds the features for dates[i]; rows before
    the 30-day spinup are NaN. Any drift from the scalar method would be a
    train/serve skew, so run parity_check() after changing either.
    """
    n = len(cs.precip)
    dates = [d for d in cs.idx]           # insertion order == chronological
    X = np.full((n, len(FEATURES)), np.nan)
    col = {f: j for j, f in enumerate(FEATURES)}
    spin = max(WINDOWS)

    for w in WINDOWS:
        ws = cs.window_sums(w)
        fin = np.isfinite(ws)
        X[:, col[f"precip_{w}d"]] = ws

        hist = np.sort(ws[fin])
        p = np.full(n, np.nan)
        if hist.size:
            p[fin] = np.searchsorted(hist, ws[fin], side="right") / hist.size
        X[:, col[f"precip_{w}d_pctl"]] = p

        seas = np.full(n, np.nan)
        for t in np.unique(cs.doy):
            m = fin & (np.abs(cs.doy - t) <= 15)
            hs = np.sort(ws[m])
            if hs.size >= 30:
                idx = np.where(cs.doy == t)[0]
                seas[idx] = np.searchsorted(hs, ws[idx], side="right") / hs.size
        X[:, col[f"precip_{w}d_pctl_seasonal"]] = seas

    ws30 = cs.window_sums(30)
    mean30 = np.nanmean(ws30)
    if np.isfinite(mean30) and mean30 > 0:
        X[:, col["precip_30d_over_climo_mean"]] = ws30 / mean30

    X[:spin, :] = np.nan              # features() refuses these indices
    return dates, X


def parity_check(cs: CellSeries, dates: list[str], X: np.ndarray,
                 n_probe: int = 6) -> None:
    rng = random.Random(SEED)
    spin = max(WINDOWS)
    for _ in range(n_probe):
        i = rng.randrange(spin, len(dates))
        ref = cs.features(dates[i])
        assert ref is not None
        for f, v in ref.items():
            got = X[i, FEATURES.index(f)]
            if np.isnan(v) and np.isnan(got):
                continue
            if abs(v - got) > 1e-9:
                raise AssertionError(
                    f"parity failure at {dates[i]} {f}: scalar={v} vector={got}")


# ------------------------------------------------------- temporal freezing ---

def load_pre_split() -> pd.DataFrame:
    """Training rows fully contained in <= SPLIT_YEAR.

    A stratum survives if its case is pre-split and at least one of its
    controls is too; controls dated after the split are dropped so no
    post-split weather (even as a 'nothing happened' example) reaches the
    model the hindcast then grades.
    """
    df = pd.read_csv(PROCESSED / "trigger.csv")
    df["year"] = df["date"].str[:4].astype(int)
    # a stratum can hold duplicate case rows (same event reported twice) --
    # one year per stratum is all the split needs
    case_year = (df[df.label == 1].groupby("stratum")["year"].first())
    df = df[df["year"] <= SPLIT_YEAR]
    df = df[df["stratum"].map(case_year).le(SPLIT_YEAR)]
    ok = df.groupby("stratum")["label"].agg(["max", "min"])
    keep = ok[(ok["max"] == 1) & (ok["min"] == 0)].index
    return df[df["stratum"].isin(keep)].reset_index(drop=True)


def freeze_model(df: pd.DataFrame, n_tune: int = 15):
    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats].to_numpy("float64")
    y = df["label"].to_numpy()
    g = df["wx_cell"].to_numpy()

    def fp(params):
        def _f(Xtr, ytr, Xte):
            return LGBMClassifier(**params).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        return _f

    rng = random.Random(SEED)
    best = (None, -1.0)
    for i in range(1, n_tune + 1):
        p = dict(BASE_PARAMS)
        p.update({k: rng.choice(v) for k, v in SEARCH.items()})
        folds, _ = spatial_cv(X, y, g, fp(p), n_splits=5)
        pr = np.mean([f["pr_auc"] for f in folds if "pr_auc" in f])
        tag = ""
        if pr > best[1]:
            best, tag = (p, pr), "  <-- best"
        print(f"    [{i:>2}] pre-{SPLIT_YEAR+1} PR-AUC {pr:.4f}{tag}", flush=True)
    params = best[0] or dict(BASE_PARAMS)

    folds, oof = spatial_cv(X, y, g, fp(params), n_splits=5)
    m = np.isfinite(oof)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[m], y[m])
    op = threshold_for_recall(y[m], iso.predict(oof[m]), 0.80)
    model = LGBMClassifier(**params).fit(X, y)
    return model, iso, float(op["threshold"]), feats, params


# --------------------------------------------------------------- hindcast ---

def replay(model, iso, feats):
    """Score every day 2016-2024 at every cached cell. Returns long-form df."""
    reg = region()
    ev = load_reports(reg["bbox"], ACC_WEATHER, need_date=True)
    ev = [e for e in ev if e["date"] >= nasapower.START]
    cells = sorted({nasapower.cell(e["lat"], e["lon"]) for e in ev})

    event_days = {}
    for e in ev:
        c = nasapower.cell(e["lat"], e["lon"])
        event_days.setdefault(c, set()).add(e["date"])

    frames = []
    checked = False
    for c in cells:
        raw = nasapower.fetch_cell(*c)
        if raw is None:
            continue
        cs = CellSeries(raw)
        dates, X = features_all(cs)
        if not checked:
            parity_check(cs, dates, X)
            print("  parity check vs training-time features: OK", flush=True)
            checked = True
        sel = [i for i, d in enumerate(dates) if d[:4] > str(SPLIT_YEAR)]
        Xs = X[sel][:, [FEATURES.index(f) for f in feats]]
        good = ~np.isnan(Xs).all(axis=1)
        p = iso.predict(model.predict_proba(Xs)[:, 1])
        frames.append(pd.DataFrame({
            "cell": f"{c[0]}_{c[1]}",
            "date": [dates[i] for i in sel],
            "p": np.where(good, p, np.nan),
            "rule": Xs[:, feats.index("precip_3d_pctl_seasonal")],
            "event": [dates[i] in event_days.get(c, ()) for i in sel],
        }))
    return pd.concat(frames, ignore_index=True)


def score_alarms(df: pd.DataFrame, alarm: np.ndarray) -> dict:
    """POD/FAR/alarm-rate for a boolean alarm vector, strict and +/-1 day."""
    d = df.assign(alarm=alarm)
    # +/-1 day neighbourhoods within each cell (rows are chronological per cell)
    g = d.groupby("cell", sort=False)
    near_event = (g["event"].shift(1, fill_value=False) | d["event"]
                  | g["event"].shift(-1, fill_value=False))
    near_alarm = (g["alarm"].shift(1, fill_value=False) | d["alarm"]
                  | g["alarm"].shift(-1, fill_value=False))
    n_ev = int(d["event"].sum())
    n_al = int(d["alarm"].sum())
    return {
        "n_event_days": n_ev,
        "n_alarm_days": n_al,
        "alarm_rate": n_al / len(d),
        "pod_strict": float((d["event"] & d["alarm"]).sum() / n_ev) if n_ev else None,
        "pod_1d": float((d["event"] & near_alarm).sum() / n_ev) if n_ev else None,
        "far_strict": float(1 - (d["alarm"] & d["event"]).sum() / n_al) if n_al else None,
        "far_1d": float(1 - (d["alarm"] & near_event).sum() / n_al) if n_al else None,
    }


def sweep(df: pd.DataFrame, score_col: str, n: int = 40) -> list[dict]:
    v = df[score_col].to_numpy()
    fin = np.isfinite(v)
    qs = np.quantile(v[fin], np.linspace(0.80, 0.9995, n))
    out = []
    for t in np.unique(qs):
        s = score_alarms(df, fin & (v >= t))
        s["threshold"] = float(t)
        out.append(s)
    return out


def main():
    print(f"=== freezing model on data through {SPLIT_YEAR} ===")
    tr = load_pre_split()
    print(f"  pre-split training rows: {len(tr):,} "
          f"({int(tr.label.sum()):,} cases, {tr.stratum.nunique():,} strata, "
          f"{tr.wx_cell.nunique()} cells)")
    model, iso, thr, feats, params = freeze_model(tr)
    print(f"  frozen operating threshold (80% recall on pre-split OOF): {thr:.3f}")

    print(f"\n=== replaying {SPLIT_YEAR+1}-2024 ===")
    grid = replay(model, iso, feats)
    n_days = grid.groupby("cell").size().iloc[0]
    print(f"  {grid.cell.nunique()} cells x {n_days:,} days = {len(grid):,} cell-days, "
          f"{int(grid.event.sum())} reported event-days "
          f"(base rate {grid.event.mean():.5f})")

    fin = np.isfinite(grid["p"].to_numpy())
    at_op = score_alarms(grid, fin & (grid["p"].to_numpy() >= thr))
    print(f"\n  at frozen threshold {thr:.3f}:")
    for k, v in at_op.items():
        print(f"    {k:<14} {v if not isinstance(v, float) else round(v, 4)}")

    print("\n  sweeping model + baseline rule...")
    curve_model = sweep(grid, "p")
    curve_rule = sweep(grid, "rule")

    # model-vs-rule at matched alarm rates (the value-added table)
    def pod_at(curve, rate):
        c = min(curve, key=lambda s: abs(s["alarm_rate"] - rate))
        return c["pod_1d"], c["alarm_rate"]
    matched = []
    for rate in (0.01, 0.02, 0.05, 0.10):
        pm, rm = pod_at(curve_model, rate)
        pr_, rr = pod_at(curve_rule, rate)
        matched.append({"target_alarm_rate": rate,
                        "model_pod_1d": pm, "model_rate": rm,
                        "rule_pod_1d": pr_, "rule_rate": rr})
        print(f"    ~{rate:.0%} alarm days: model catches {pm:.0%}, "
              f"3d-seasonal-percentile rule catches {pr_:.0%}")

    yearly = []
    g = grid.assign(alarm=fin & (grid["p"].to_numpy() >= thr),
                    year=grid["date"].str[:4])
    for yr, gy in g.groupby("year"):
        s = score_alarms(gy.reset_index(drop=True),
                         gy["alarm"].to_numpy())
        yearly.append({"year": yr, "events": s["n_event_days"],
                       "pod_1d": s["pod_1d"],
                       "alarms_per_cell": s["n_alarm_days"] / gy.cell.nunique()})

    out = {
        "name": "hindcast-trigger",
        "layer": "trigger",
        "protocol": (f"trained/tuned/calibrated/thresholded on <= {SPLIT_YEAR} only; "
                     f"replayed {SPLIT_YEAR+1}-2024 on every day at every cell"),
        "params": params,
        "operating_threshold": thr,
        "grid": {"cells": int(grid.cell.nunique()), "cell_days": int(len(grid)),
                 "event_days": int(grid.event.sum()),
                 "base_rate": float(grid.event.mean())},
        "at_operating_point": at_op,
        "matched_alarm_rate": matched,
        "curve_model": curve_model,
        "curve_rule": curve_rule,
        "yearly": yearly,
        "caveats": [
            "events are reported events; FAR is an upper bound",
            "cells are places with >=1 historical report; silent terrain unmeasured",
            "+/-1 day tolerance reported because report dates can lag occurrence",
        ],
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / "hindcast-trigger.json"
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
