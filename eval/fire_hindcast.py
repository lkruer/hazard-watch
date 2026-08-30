"""Prospective hindcast for the fire-danger layer: what does OPERATING it feel like?

D20 measured the fire layer the way it was built -- case-crossover, one fire day
against four season-matched control days at the same spot, base rate 0.20. That
answers "does weather separate a fire day from a quiet day here", and the answer
was yes on two continents (ROC 0.76 US / 0.81 Canada, 0.77 cold transfer). It
does not answer the operational question, because deployment does not get handed
five candidate days -- it scores EVERY day at EVERY cell, where a reported
large-fire day is about 1.6-in-1,000. This module asks D14's question of the
fire layer, with D14's protocol:

  1. Freeze everything on data through 2015: keep only strata whose CASE is
     pre-2016, drop controls dated after 2015 as well, re-tune (light random
     search, grouped by weather cell), fit, and isotonically calibrate on
     pre-2016 out-of-fold predictions. Nothing from 2016+ touches the model.
  2. Replay forward on the 540 labelled fire-weather cells, each to the end of
     its own label record: US cells 2016-2020 (FPA-FOD 6th ed. stops at fire
     year 2020), Canada cells 2016-2024 (NFDB runs past 2025, so the binding
     limit there is the POWER cache, which ends 2024-12-31).
  3. Score the alarms the way forecast verification does: POD (strict and
     +/-1 day), FAR, alarm rate -- at alarm budgets of 2/5/10% of days, with
     the thresholds taken from the replay grid's OWN score quantiles, because
     D14's central finding was that a threshold only means something on the
     distribution it will actually score.
  4. Compare against the one-line rule "alarm when vpd_pctl_seasonal is high".
     D20 showed VPD alone carries 0.72-0.74 ROC of the model's 0.78. Whether
     the 14-feature model beats a rule anyone could write in a spreadsheet is
     measured here prospectively, not assumed.
  5. Audit the threshold already recorded for deployment
     (serve/thresholds.json -> fire -> 5% budget, 0.4028): what alarm rate and
     POD does that number actually deliver, forward in time?

Honesty notes baked into the output:
  - Labels are REPORTED fires (FPA-FOD >=100 acres, NFDB >=100 ha). An alarm on
    a dangerous day where nothing was reported counts as false, so FAR here is
    an upper bound and POD is measured on reported fires only.
  - The 540 cells are cells that burn often enough to have supplied >=4 fires
    to D20's strata. Behaviour in rarely-burning terrain is not measured.
  - A +/-1 day tolerance is reported alongside strict matching: discovery date
    is when a fire was FOUND, which can lag ignition, and the danger state that
    made it dangerous is a multi-day condition, not a single midnight-to-
    midnight box.
  - The seasonal percentiles normalise against the cell's whole 2004-2024
    record, matching FireCellSeries.features() exactly (a train/serve skew
    would be worse). That is label-free climatology, but it does mean the
    *normalisation* has seen the future; see climatology_leak in the output.
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
from eval.fire_validate import ca_fires, us_fires  # noqa: E402
from models.train import BASE_PARAMS, SEARCH  # noqa: E402
from pipelines import fireweather as fw  # noqa: E402
from pipelines.fireweather import FEATURES, WINDOWS, FireCellSeries  # noqa: E402

SPLIT_YEAR = 2015          # train <= this, evaluate strictly after
SEED = 17
SPINUP = 365               # KBDI spin-up year; features() refuses i < 365
BUDGETS = (0.02, 0.05, 0.10)
RULE_FEATURE = "vpd_pctl_seasonal"
RUNS = ROOT / "models" / "runs"

# Label records end where they end; each domain is replayed to its own limit.
# FPA-FOD 6th ed. carries fire years 1992-2020. NFDB point data runs past 2025,
# so Canada is capped by the POWER weather cache instead (fw.END).
LABEL_END = {"us": "2020-12-31", "canada": fw.END}
REPLAY_START = f"{SPLIT_YEAR + 1}-01-01"


def implausible(cell_id: str) -> bool:
    """Null-island guard.

    NFDB carries 6 Parks Canada fires >= 100 ha with LATITUDE = LONGITUDE = 0.0
    -- missing coordinates written as zeros, not NaN, so ca_fires()'s dropna
    does not catch them. Four of them share a "cell", which was enough for D20's
    build_strata to accept (0.0, 0.0) as a dense cell and pair Canadian fire
    labels with equatorial-Atlantic weather. Excluded from both the frozen
    training set and the replay grid; the count is reported in the run JSON
    rather than silently swallowed.
    """
    la, lo = (float(v) for v in cell_id.split("_"))
    return abs(la) < 1.0 and abs(lo) < 1.0


# ---------------------------------------------------- vectorized features ---

def _seasonal_pctl(arr: np.ndarray, doy: np.ndarray,
                   win: dict[int, np.ndarray]) -> np.ndarray:
    """Vector form of FireCellSeries._pctl_seasonal over every index.

    Mirrors the scalar method exactly, including two easy-to-miss behaviours:
      - fewer than 30 finite season-matched samples -> NaN;
      - a NaN value scores 0.0, not NaN, because the scalar computes
        (hs <= nan).mean(), and every comparison against NaN is False.
    """
    n = arr.size
    fin = np.isfinite(arr)
    out = np.full(n, np.nan)
    for t, idx in win.items():
        m = idx[fin[idx]]
        if m.size < 30:
            continue
        hs = np.sort(arr[m])
        here = np.where(doy == t)[0]
        v = arr[here]
        p = np.searchsorted(hs, v, side="right") / hs.size
        out[here] = np.where(np.isnan(v), 0.0, p)
    return out


def _alltime_pctl(arr: np.ndarray) -> np.ndarray:
    """Vector form of FireCellSeries._pctl (all-time, same NaN convention)."""
    n = arr.size
    fin = np.isfinite(arr)
    if not fin.any():
        return np.full(n, np.nan)
    hs = np.sort(arr[fin])
    p = np.searchsorted(hs, arr, side="right") / hs.size
    return np.where(np.isnan(arr), 0.0, p)


def features_all(cs: FireCellSeries) -> tuple[list[str], np.ndarray]:
    """Whole-series feature matrix, exactly mirroring FireCellSeries.features().

    Returns (dates, X) where X[i] holds the features for dates[i]; rows inside
    the KBDI spin-up year are NaN, since features() refuses those indices. Any
    drift from the scalar method is a train/serve skew, so parity_check() runs
    against .features() on real cells every time this is used.
    """
    dates = list(cs.idx)                   # insertion order == chronological
    n = len(dates)
    X = np.full((n, len(FEATURES)), np.nan)
    col = {f: j for j, f in enumerate(FEATURES)}

    # season windows are pure functions of doy: build the index lists once and
    # reuse them for all seven percentile series in this cell
    win = {int(t): np.where(np.abs(cs.doy - t) <= 15)[0]
           for t in np.unique(cs.doy)}

    X[:, col["kbdi"]] = cs.kbdi
    X[:, col["kbdi_pctl"]] = _alltime_pctl(cs.kbdi)
    X[:, col["kbdi_pctl_seasonal"]] = _seasonal_pctl(cs.kbdi, cs.doy, win)
    X[:, col["vpd_kpa"]] = cs.vpd
    X[:, col["vpd_pctl_seasonal"]] = _seasonal_pctl(cs.vpd, cs.doy, win)
    X[:, col["tmax_c"]] = cs.tmax
    X[:, col["tmax_pctl_seasonal"]] = _seasonal_pctl(cs.tmax, cs.doy, win)
    X[:, col["rh_pct"]] = cs.rh
    X[:, col["rh_pctl_seasonal"]] = _seasonal_pctl(cs.rh, cs.doy, win)
    X[:, col["ws_ms"]] = cs.ws
    X[:, col["ws_pctl_seasonal"]] = _seasonal_pctl(cs.ws, cs.doy, win)
    X[:, col["days_since_rain"]] = cs.dsr
    for w in WINDOWS:
        X[:, col[f"precip_{w}d_pctl_seasonal"]] = _seasonal_pctl(
            cs._wsum(w), cs.doy, win)

    X[:SPINUP, :] = np.nan
    return dates, X


def parity_check(cs: FireCellSeries, dates: list[str], X: np.ndarray,
                 n_probe: int = 6, tag: str = "") -> int:
    """Fail loudly if the vector path drifts from the scalar path."""
    rng = random.Random(SEED)
    for _ in range(n_probe):
        i = rng.randrange(SPINUP, len(dates))
        ref = cs.features(dates[i])
        assert ref is not None
        for f, v in ref.items():
            got = X[i, FEATURES.index(f)]
            if np.isnan(v) and np.isnan(got):
                continue
            if not (abs(v - got) <= 1e-9):
                raise AssertionError(
                    f"parity failure {tag} at {dates[i]} {f}: "
                    f"scalar={v!r} vector={got!r}")
    return n_probe


# ------------------------------------------------------- temporal freezing ---

def load_pre_split() -> tuple[pd.DataFrame, dict]:
    """Training rows fully contained in <= SPLIT_YEAR.

    A stratum survives if its case is pre-split and at least one of its controls
    is too; controls dated after the split are dropped so no post-split weather
    -- even as a 'nothing burned' example -- reaches the model the hindcast then
    grades. A stratum can in principle hold duplicate case rows (two fires at
    one coordinate on one day collide on the stratum id), so the case year is
    taken per stratum, not per row.
    """
    df = pd.read_csv(PROCESSED / "fire_trigger.csv")
    df["year"] = df["date"].str[:4].astype(int)
    n_all = len(df)
    junk = sorted({c for c in df.wx_cell.unique() if implausible(c)})
    n_junk = int(df.wx_cell.isin(junk).sum())
    df = df[~df.wx_cell.isin(junk)]
    case_year = df[df.label == 1].groupby("stratum")["year"].first()
    df = df[df["year"] <= SPLIT_YEAR]
    df = df[df["stratum"].map(case_year).le(SPLIT_YEAR)]
    ok = df.groupby("stratum")["label"].agg(["max", "min"])
    keep = ok[(ok["max"] == 1) & (ok["min"] == 0)].index
    lost = int(len(ok) - len(keep))
    out = df[df["stratum"].isin(keep)].reset_index(drop=True)
    info = {
        "rows_all": int(n_all),
        "rows_pre_split": int(len(out)),
        "strata_pre_split": int(out.stratum.nunique()),
        "cases_pre_split": int(out.label.sum()),
        "strata_dropped_no_control_left": lost,
        "cells": int(out.wx_cell.nunique()),
        "by_domain": {k: int(v) for k, v in
                      out.groupby("domain")["stratum"].nunique().items()},
        "null_island_cells_excluded": junk,
        "null_island_rows_excluded": n_junk,
        # not D20's 0.20: dropping post-split controls thins the control side,
        # so the frozen model trains against a richer case fraction
        "base_rate": float(out.label.mean()),
    }
    return out, info


def freeze_model(df: pd.DataFrame, n_tune: int = 10):
    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats].to_numpy("float64")
    y = df["label"].to_numpy()
    g = df["wx_cell"].to_numpy()

    def fp(params):
        def _f(Xtr, ytr, Xte):
            return LGBMClassifier(**params).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        return _f

    rng = random.Random(SEED)
    base_folds, _ = spatial_cv(X, y, g, fp(dict(BASE_PARAMS)), n_splits=5)
    base_pr = float(np.mean([f["pr_auc"] for f in base_folds if "pr_auc" in f]))
    print(f"    [base] pre-{SPLIT_YEAR+1} PR-AUC {base_pr:.4f}", flush=True)
    best = (dict(BASE_PARAMS), base_pr)
    for i in range(1, n_tune + 1):
        p = dict(BASE_PARAMS)
        p.update({k: rng.choice(v) for k, v in SEARCH.items()})
        folds, _ = spatial_cv(X, y, g, fp(p), n_splits=5)
        pr = float(np.mean([f["pr_auc"] for f in folds if "pr_auc" in f]))
        tag = ""
        if pr > best[1]:
            best, tag = (p, pr), "  <-- best"
        print(f"    [{i:>2}] pre-{SPLIT_YEAR+1} PR-AUC {pr:.4f}{tag}", flush=True)
    params, cv_pr = best

    folds, oof = spatial_cv(X, y, g, fp(params), n_splits=5)
    m = np.isfinite(oof)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[m], y[m])
    trained_thr = threshold_for_recall(y[m], iso.predict(oof[m]), 0.80)
    model = LGBMClassifier(**params).fit(X, y)
    return model, iso, feats, params, cv_pr, trained_thr


# --------------------------------------------------------------- hindcast ---

def grid_cells() -> dict[str, str]:
    """cell id -> domain, for the labelled cells only, cache-backed only."""
    df = pd.read_csv(PROCESSED / "fire_trigger.csv")
    dom = df.groupby("wx_cell")["domain"].first().to_dict()
    out = {}
    for c, d in dom.items():
        if implausible(c):
            continue
        la, lo = (float(v) for v in c.split("_"))
        if fw._path(la, lo).exists():        # never trigger a network fetch
            out[c] = d
    return out


def event_days(cells: dict[str, str]) -> tuple[dict[str, set], dict]:
    """Reported fire discovery days per cell, from the D20 label loaders."""
    ev: dict[str, set] = {}
    stats = {}
    for tag, loader in (("us", us_fires), ("canada", ca_fires)):
        want = {c for c, d in cells.items() if d == tag}
        df = loader()
        cid = [f"{a}_{b}" for a, b in
               (fw.cell(la, lo) for la, lo in zip(df["lat"], df["lon"]))]
        df = df.assign(cell=cid)
        df = df[df["cell"].isin(want)]
        df = df[(df["date"] >= REPLAY_START) & (df["date"] <= LABEL_END[tag])]
        for c, d in zip(df["cell"], df["date"]):
            ev.setdefault(c, set()).add(d)
        stats[tag] = {"fires_in_window": int(len(df)),
                      "event_cell_days": int(sum(
                          len(v) for c, v in ev.items() if cells[c] == tag)),
                      "cells_with_events": int(sum(
                          1 for c in ev if cells[c] == tag)),
                      "window": [REPLAY_START, LABEL_END[tag]]}
    return ev, stats


def replay(model, iso, feats, prod=None) -> tuple[pd.DataFrame, dict]:
    """Score every valid day at every labelled cell. Memory stays flat: one
    cell's series is built, scored, reduced to compact arrays, and dropped."""
    cells = grid_cells()
    ev, ev_stats = event_days(cells)
    order = sorted(cells)
    cidx = [FEATURES.index(f) for f in feats]
    rule_j = FEATURES.index(RULE_FEATURE)

    probe_at = {0, len(order) // 3, 2 * len(order) // 3, len(order) - 1}
    parity_done = 0
    nan_any = 0
    cols = {k: [] for k in ("cell", "year", "dom", "p", "prod", "rule", "event")}
    codes = {c: i for i, c in enumerate(order)}

    for n, c in enumerate(order):
        la, lo = (float(v) for v in c.split("_"))
        raw = fw.fetch_cell(la, lo)           # cache-only: existence pre-checked
        if raw is None:
            continue
        cs = FireCellSeries(raw)
        dates, X = features_all(cs)
        if n in probe_at:
            parity_done += parity_check(cs, dates, X, tag=f"cell {c}")

        end = LABEL_END[cells[c]]
        sel = np.array([i for i, d in enumerate(dates)
                        if i >= SPINUP and REPLAY_START <= d <= end])
        if not sel.size:
            continue
        Xs = X[sel][:, cidx]
        valid = ~np.isnan(Xs).all(axis=1)
        nan_any += int(np.isnan(Xs).any(axis=1).sum())
        p = iso.predict(model.predict_proba(Xs)[:, 1])
        cols["p"].append(np.where(valid, p, np.nan).astype("float32"))
        if prod is not None:
            pp = prod["calibrator"].predict(
                prod["model"].predict_proba(Xs)[:, 1])
            cols["prod"].append(np.where(valid, pp, np.nan).astype("float32"))
        cols["rule"].append(Xs[:, feats.index(RULE_FEATURE)].astype("float32"))
        seen = ev.get(c, ())
        cols["event"].append(np.array([dates[i] in seen for i in sel]))
        cols["year"].append(np.array([int(dates[i][:4]) for i in sel],
                                     dtype="int16"))
        cols["cell"].append(np.full(sel.size, codes[c], dtype="int16"))
        cols["dom"].append(np.full(sel.size, cells[c] == "us", dtype=bool))
        del cs, X, Xs
        fw._CACHE.clear()                     # keep memory flat across 540 cells
        if (n + 1) % 60 == 0:
            print(f"    {n+1}/{len(order)} cells", flush=True)

    if not parity_done:
        raise AssertionError("parity check never ran -- refusing to report")
    print(f"  parity check vs FireCellSeries.features(): OK "
          f"({parity_done} probes on {len(probe_at)} cells)", flush=True)
    grid = pd.DataFrame({
        "cell": np.concatenate(cols["cell"]),
        "year": np.concatenate(cols["year"]),
        "us": np.concatenate(cols["dom"]),
        "p": np.concatenate(cols["p"]),
        "rule": np.concatenate(cols["rule"]),
        "event": np.concatenate(cols["event"]),
    })
    if prod is not None:
        grid["prod"] = np.concatenate(cols["prod"])
    ev_stats["rows_with_any_nan_feature"] = nan_any
    ev_stats["rule_feature"] = RULE_FEATURE
    return grid, ev_stats


# ---------------------------------------------------------------- scoring ---

def score_alarms(df: pd.DataFrame, alarm: np.ndarray) -> dict:
    """POD/FAR/alarm-rate for a boolean alarm vector, strict and +/-1 day."""
    d = df.assign(alarm=alarm)
    g = d.groupby("cell", sort=False)
    near_event = (g["event"].shift(1, fill_value=False) | d["event"]
                  | g["event"].shift(-1, fill_value=False))
    near_alarm = (g["alarm"].shift(1, fill_value=False) | d["alarm"]
                  | g["alarm"].shift(-1, fill_value=False))
    n_ev = int(d["event"].sum())
    n_al = int(d["alarm"].sum())
    hit_s = int((d["event"] & d["alarm"]).sum())
    hit_1 = int((d["event"] & near_alarm).sum())
    return {
        "n_event_days": n_ev,
        "n_alarm_days": n_al,
        "alarm_rate": n_al / len(d) if len(d) else None,
        # every cell contributes one row per day, so days-per-cell-per-year is
        # just the alarm rate on a calendar year
        "alarms_per_cell_year": 365.25 * n_al / len(d) if len(d) else None,
        "n_hits_strict": hit_s,
        "n_hits_1d": hit_1,
        "pod_strict": hit_s / n_ev if n_ev else None,
        "pod_1d": hit_1 / n_ev if n_ev else None,
        "far_strict": 1 - hit_s / n_al if n_al else None,
        "far_1d": float(1 - (d["alarm"] & near_event).sum() / n_al) if n_al else None,
    }


def at_threshold(grid: pd.DataFrame, col: str, thr: float) -> dict:
    v = grid[col].to_numpy()
    fin = np.isfinite(v)
    s = score_alarms(grid, fin & (v >= thr))
    s["threshold"] = float(thr)
    return s


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


def budget_threshold(grid: pd.DataFrame, col: str, budget: float) -> float:
    v = grid[col].to_numpy()
    return float(np.quantile(v[np.isfinite(v)], 1 - budget))


def main():
    t0 = dt.datetime.now()
    print(f"=== freezing fire model on data through {SPLIT_YEAR} ===")
    tr, info = load_pre_split()
    print(f"  pre-split training rows: {info['rows_pre_split']:,} "
          f"({info['cases_pre_split']:,} cases, "
          f"{info['strata_pre_split']:,} strata, {info['cells']} cells; "
          f"{info['strata_dropped_no_control_left']} strata dropped for having "
          f"no pre-split control left)")
    print(f"  strata by domain: {info['by_domain']}")
    if info["null_island_cells_excluded"]:
        print(f"  data quality: dropped "
              f"{info['null_island_rows_excluded']} rows at null-island cells "
              f"{info['null_island_cells_excluded']} (NFDB lat=lon=0.0)")
    model, iso, feats, params, cv_pr, trained_thr = freeze_model(tr)
    print(f"  frozen pre-{SPLIT_YEAR+1} CV PR-AUC {cv_pr:.4f}")
    print(f"  case-control 80%-recall threshold (the D14 trap): "
          f"{trained_thr['threshold']:.4f} "
          f"(alarm rate {trained_thr['alarm_rate']:.1%} on case-control rows)")

    prod = None
    pk = ROOT / "models" / "artifacts" / "fire_trigger.pkl"
    if pk.exists():
        import pickle
        with pk.open("rb") as fh:
            prod = pickle.load(fh)
        if list(prod["features"]) != list(feats):
            print("  production artifact feature mismatch -- skipping its audit")
            prod = None

    print(f"\n=== replaying {REPLAY_START[:4]}+ "
          f"(us -> {LABEL_END['us'][:4]}, canada -> {LABEL_END['canada'][:4]}) ===")
    grid, ev_stats = replay(model, iso, feats, prod=prod)
    per_cell = grid.groupby("cell").size()
    print(f"  {grid.cell.nunique()} cells x {per_cell.min():,}-{per_cell.max():,} "
          f"days = {len(grid):,} cell-days, "
          f"{int(grid.event.sum()):,} reported event-days "
          f"(base rate {grid.event.mean():.5f})")
    for tag in ("us", "canada"):
        sub = grid[grid.us == (tag == "us")]
        print(f"    [{tag}] {sub.cell.nunique()} cells, {len(sub):,} cell-days, "
              f"{int(sub.event.sum()):,} events "
              f"(base rate {sub.event.mean():.5f})")

    # ---- the honest operating menu: budgets priced on the replay grid itself
    print("\n=== operating menu (thresholds from the replay grid's own "
          "score quantiles) ===")
    at_budget = []
    for b in BUDGETS:
        tm = budget_threshold(grid, "p", b)
        tr_ = budget_threshold(grid, "rule", b)
        sm = at_threshold(grid, "p", tm)
        sr = at_threshold(grid, "rule", tr_)
        row = {"budget": b, "model": sm, "rule": sr}
        for tag in ("us", "canada"):
            sub = grid[grid.us == (tag == "us")].reset_index(drop=True)
            # at the POOLED threshold: shows how unevenly one continental
            # threshold spends the alarm budget between the two domains
            row[tag] = at_threshold(sub, "p", tm)
            # and at the domain's OWN quantile: what a per-domain threshold buys
            row[f"{tag}_own"] = at_threshold(sub, "p",
                                             budget_threshold(sub, "p", b))
        at_budget.append(row)
        print(f"  {b:.0%} budget  thr {tm:.4f}  "
              f"alarm {sm['alarm_rate']:.2%} ({sm['alarms_per_cell_year']:.1f} "
              f"days/cell/yr)  POD1d {sm['pod_1d']:.1%} "
              f"({sm['n_hits_1d']}/{sm['n_event_days']})   |   "
              f"VPD rule POD1d {sr['pod_1d']:.1%}")

    # ---- what the recorded deployment threshold actually delivers
    tj = json.loads((ROOT / "serve" / "thresholds.json").read_text(encoding="utf-8"))
    rec = float(tj["fire"]["threshold"])
    rec_rows = {"threshold": rec,
                "source": "serve/thresholds.json fire.budgets['0.05']",
                "frozen_model": at_threshold(grid, "p", rec)}
    if prod is not None:
        rec_rows["production_model_in_sample"] = at_threshold(grid, "prod", rec)
    print(f"\n=== recorded deployment threshold {rec:.4f} on the prospective "
          f"grid ===")
    fm = rec_rows["frozen_model"]
    print(f"  frozen model: alarm {fm['alarm_rate']:.2%} "
          f"({fm['alarms_per_cell_year']:.1f} days/cell/yr), "
          f"POD1d {fm['pod_1d']:.1%}, FAR1d {fm['far_1d']:.4f}")
    if prod is not None:
        pm = rec_rows["production_model_in_sample"]
        print(f"  production model (IN-SAMPLE, not a prospective claim): "
              f"alarm {pm['alarm_rate']:.2%}, POD1d {pm['pod_1d']:.1%}")

    # ---- the case-control threshold, to show whether D14's trap repeats
    trap = at_threshold(grid, "p", trained_thr["threshold"])
    print(f"\n=== D14 trap check: case-control 80%-recall threshold "
          f"{trained_thr['threshold']:.4f} ===")
    print(f"  would alarm on {trap['alarm_rate']:.1%} of all real days "
          f"({trap['alarms_per_cell_year']:.0f} days/cell/yr), "
          f"POD1d {trap['pod_1d']:.1%}")

    print("\n  sweeping model + VPD rule...", flush=True)
    curve_model = sweep(grid, "p")
    curve_rule = sweep(grid, "rule")

    yearly = []
    thr5 = budget_threshold(grid, "p", 0.05)
    for scope, sub in (("all", grid),
                       ("us", grid[grid.us]), ("canada", grid[~grid.us])):
        for yr, gy in sub.groupby("year"):
            gy = gy.reset_index(drop=True)
            s = at_threshold(gy, "p", thr5)
            yearly.append({"scope": scope, "year": int(yr),
                           "cells": int(gy.cell.nunique()),
                           "cell_days": int(len(gy)),
                           "events": s["n_event_days"],
                           "pod_strict": s["pod_strict"], "pod_1d": s["pod_1d"],
                           "alarm_rate": s["alarm_rate"],
                           "alarms_per_cell_year": s["alarms_per_cell_year"]})

    out = {
        "name": "hindcast-fire",
        "layer": "fire",
        "status": "complete",
        "protocol": (f"trained/tuned/calibrated on <= {SPLIT_YEAR} strata only "
                     f"(controls dated after the split dropped too); replayed "
                     f"every day at every labelled cell from {REPLAY_START}, "
                     f"us to {LABEL_END['us']} (FPA-FOD 6th ed. ends at fire "
                     f"year 2020), canada to {LABEL_END['canada']} (NFDB runs "
                     f"past 2025; the POWER weather cache is the binding limit)"),
        "split_year": SPLIT_YEAR,
        "params": params,
        "features": feats,
        "training": {**info, "pre_split_cv_pr_auc": cv_pr,
                     "case_control_80pct_recall_threshold": trained_thr},
        "labels": {"us": "FPA-FOD 6th ed. discoveries >= 100 acres (public domain)",
                   "canada": "NFDB point records >= 100 ha (open)",
                   **ev_stats},
        "grid": {"cells": int(grid.cell.nunique()),
                 "cell_days": int(len(grid)),
                 "event_days": int(grid.event.sum()),
                 "base_rate": float(grid.event.mean()),
                 "us": {"cells": int(grid[grid.us].cell.nunique()),
                        "cell_days": int(grid.us.sum()),
                        "event_days": int(grid[grid.us].event.sum()),
                        "base_rate": float(grid[grid.us].event.mean())},
                 "canada": {"cells": int(grid[~grid.us].cell.nunique()),
                            "cell_days": int((~grid.us).sum()),
                            "event_days": int(grid[~grid.us].event.sum()),
                            "base_rate": float(grid[~grid.us].event.mean())}},
        "at_budget": at_budget,
        "recorded_threshold_audit": rec_rows,
        "case_control_threshold_on_grid": trap,
        "curve_model": curve_model,
        "curve_rule": curve_rule,
        "yearly": yearly,
        "caveats": [
            "labels are REPORTED fires (FPA-FOD >=100 acres, NFDB >=100 ha); an "
            "alarm on a dangerous day with no reported fire counts as false, so "
            "FAR is an upper bound and POD is measured on reported fires only",
            "the 540 cells each supplied >=4 fires to the D20 strata, so this is "
            "measured on frequently-burning terrain; rarely-burning cells are "
            "not represented",
            "+/-1 day tolerance is reported because a discovery date is when a "
            "fire was found, not when it started, and fire danger is a "
            "multi-day state rather than a midnight-to-midnight box",
            "us and canada replay windows differ (2016-2020 vs 2016-2024), so "
            "pooled per-year rows after 2020 are Canada-only -- see yearly[] "
            "scope field",
            "seasonal percentiles normalise against the cell's full 2004-2024 "
            "record, matching FireCellSeries.features() exactly; the model, "
            "calibration and thresholds are frozen pre-2016 but the "
            "climatology denominator is not (label-free, but stated)",
            "no fuel or vegetation layer (D20): danger here is weather-only",
        ],
        "data_quality": {
            "null_island": (
                "NFDB contains 6 Parks Canada fires >= 100 ha recorded at "
                "LATITUDE = LONGITUDE = 0.0 (missing coordinates written as "
                "zeros; ca_fires()'s dropna does not catch them). Four fell in "
                "one 0.5-degree cell, so D20's build_strata accepted (0.0, 0.0) "
                "as a dense cell: 4 of its 2,160 strata (20 rows) pair Canadian "
                "fire labels with equatorial-Atlantic weather. Excluded here "
                "from training and grid; upstream fix belongs in "
                "eval/fire_validate.ca_fires()."),
            "cells_excluded": info["null_island_cells_excluded"],
            "rows_excluded_from_strata": info["null_island_rows_excluded"],
        },
        "climatology_leak": (
            "features_all mirrors FireCellSeries.features(), which computes "
            "seasonal percentiles against the whole cached record. Changing it "
            "for the hindcast would introduce a train/serve skew worse than the "
            "leak; parity with the deployed feature path was chosen instead."),
        "runtime_sec": (dt.datetime.now() - t0).total_seconds(),
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / "hindcast-fire.json"
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    # ------------------------------------------------------ compact summary ---
    print("\n" + "=" * 74)
    print(f"PROSPECTIVE FIRE HINDCAST -- {grid.cell.nunique()} cells, "
          f"{len(grid):,} cell-days, {int(grid.event.sum()):,} reported "
          f"event-days (base rate {grid.event.mean():.2e})")
    print("=" * 74)
    print(f"{'budget':>7} {'thresh':>8} {'days/cell/yr':>13} "
          f"{'POD strict':>11} {'POD +-1d':>9} {'caught':>12} {'VPD rule':>9}")
    for row in at_budget:
        m, r = row["model"], row["rule"]
        print(f"{row['budget']:>6.0%} {m['threshold']:>8.4f} "
              f"{m['alarms_per_cell_year']:>13.1f} "
              f"{m['pod_strict']:>10.1%} {m['pod_1d']:>9.1%} "
              f"{m['n_hits_1d']:>6}/{m['n_event_days']:<5} {r['pod_1d']:>8.1%}")
    print("-" * 74)
    print(f"{'recorded 0.4028':>22}: alarm {fm['alarm_rate']:.2%} "
          f"({fm['alarms_per_cell_year']:.1f}/cell/yr), POD+-1d {fm['pod_1d']:.1%}")
    print(f"{'case-control thresh':>22}: alarm {trap['alarm_rate']:.1%}, "
          f"POD+-1d {trap['pod_1d']:.1%}")
    print(f"{'us base rate':>22}: {grid[grid.us].event.mean():.2e}   "
          f"canada: {grid[~grid.us].event.mean():.2e}")
    for row in at_budget:
        print(f"{row['budget']:>6.0%} pooled-threshold spend: "
              f"us {row['us']['alarm_rate']:.2%} alarm / POD1d "
              f"{row['us']['pod_1d']:.1%}   canada "
              f"{row['canada']['alarm_rate']:.2%} alarm / POD1d "
              f"{row['canada']['pod_1d']:.1%}")
    try:
        shown = p.relative_to(ROOT)
    except ValueError:
        shown = p
    print(f"\nwrote {shown}  ({out['runtime_sec']/60:.1f} min)")


if __name__ == "__main__":
    main()
