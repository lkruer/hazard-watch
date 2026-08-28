"""Validate GloFAS flow percentiles against real USGS river gauges.

The flood layer's claim is narrow, per the brief: we do not model discharge --
we express GloFAS's modeled discharge as a percentile of that cell's own
seasonal record. This script checks that claim against instrumented truth:
four USGS gauges (public NWIS API, no key) spanning four river scales, from
the Columbia (~5,000 m3/s) to a Coast Range coastal river (~30 m3/s).

Two questions, in order of importance:
  1. RANK fidelity -- does the GloFAS percentile series track the gauge's
     percentile series? (Spearman rho of daily seasonal percentiles.) This is
     the only thing the flood layer actually uses.
  2. FLOOD-day hit rate -- on days the gauge sat above its own seasonal 98th
     percentile, how often was the GloFAS percentile also extreme (>= 0.95)?

Grid-to-gauge matching: a 0.05-degree river network does not sit exactly on
gauge coordinates, so within a 5x5 cell neighborhood we pick the cell whose
median discharge is closest to the gauge's median in log space -- standard
practice for coarse-grid hydrology evaluation, and recorded per gauge.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT  # noqa: E402
from pipelines.common import SESSION  # noqa: E402
from pipelines.glofas import PNW_BBOX, YEARS, DischargeStack  # noqa: E402

CFS_TO_M3S = 0.0283168
NWIS = "https://waterservices.usgs.gov/nwis/dv/"

GAUGES = [
    {"site": "14105700", "name": "Columbia @ The Dalles OR",
     "lat": 45.6075, "lon": -121.1722},
    {"site": "14211720", "name": "Willamette @ Portland OR",
     "lat": 45.5175, "lon": -122.6690},
    {"site": "12200500", "name": "Skagit @ Mount Vernon WA",
     "lat": 48.4446, "lon": -122.3346},
    {"site": "14301000", "name": "Nehalem @ Foss OR (small coastal)",
     "lat": 45.7040, "lon": -123.7554},
]


def usgs_daily(site: str) -> pd.Series:
    r = SESSION.get(NWIS, params={
        "format": "json", "sites": site,
        "startDT": f"{YEARS[0]}-01-01", "endDT": f"{YEARS[-1]}-12-31",
        "parameterCd": "00060", "statCd": "00003"}, timeout=180)
    r.raise_for_status()
    ts = r.json()["value"]["timeSeries"][0]["values"][0]["value"]
    idx, vals = [], []
    for v in ts:
        try:
            x = float(v["value"])
        except (TypeError, ValueError):
            continue
        if x < 0:                                  # -999999 = missing
            continue
        idx.append(v["dateTime"][:10])
        vals.append(x * CFS_TO_M3S)
    return pd.Series(vals, index=pd.Index(idx, name="date"), name=site)


def seasonal_pctl(s: pd.Series, window: int = 15) -> pd.Series:
    d = pd.to_datetime(s.index)
    key = d.month * 31 + d.day
    v = s.to_numpy()
    out = np.full(len(s), np.nan)
    for k in np.unique(key):
        m = np.abs(key - k) <= window
        hist = np.sort(v[m])
        ii = np.where(key == k)[0]
        if hist.size >= 30:
            out[ii] = np.searchsorted(hist, v[ii], side="right") / hist.size
    return pd.Series(out, index=s.index)


def best_cell(stack: DischargeStack, lat: float, lon: float,
              gauge_median: float):
    """Within +/-2 cells, the river cell with the closest log-median flow."""
    lat_name = next(d for d in stack.da.dims if "lat" in d.lower())
    lon_name = next(d for d in stack.da.dims if "lon" in d.lower())
    qlon = stack.q_lon(lon)
    best, best_err = None, np.inf
    for dla in np.arange(-0.10, 0.101, 0.05):
        for dlo in np.arange(-0.10, 0.101, 0.05):
            cell = stack.da.sel({lat_name: lat + dla, lon_name: qlon + dlo},
                                method="nearest")
            med = float(np.nanmedian(cell.values))
            if med < 1.0:
                continue
            err = abs(np.log10(med) - np.log10(max(gauge_median, 0.1)))
            if err < best_err:
                best_err, best = err, (float(lat + dla), float(lon + dlo), med)
    return best


def main():
    stack = DischargeStack(PNW_BBOX)
    print(f"GloFAS stack: {len(stack.dates):,} days, "
          f"{int(stack.river.sum()):,} river cells\n")
    results = []
    for g in GAUGES:
        obs = usgs_daily(g["site"])
        gm = float(obs.median())
        pick = best_cell(stack, g["lat"], g["lon"], gm)
        if pick is None:
            print(f"{g['name']}: no river cell found nearby")
            continue
        cla, clo, cmed = pick
        lat_name = next(d for d in stack.da.dims if "lat" in d.lower())
        lon_name = next(d for d in stack.da.dims if "lon" in d.lower())
        sim = pd.Series(
            stack.da.sel({lat_name: cla, lon_name: stack.q_lon(clo)},
                         method="nearest").values,
            index=pd.Index(stack.dates, name="date"))
        both = pd.concat([obs, sim], axis=1, join="inner").dropna()
        both.columns = ["obs", "sim"]
        po = seasonal_pctl(both["obs"])
        ps = seasonal_pctl(both["sim"])
        m = po.notna() & ps.notna()
        rho = float(po[m].corr(ps[m], method="spearman"))
        r_raw = float(np.corrcoef(both["obs"], both["sim"])[0, 1])
        flood = po[m] >= 0.98
        hit = float((ps[m][flood] >= 0.95).mean()) if flood.sum() else None
        res = {"gauge": g["name"], "site": g["site"],
               "n_days": int(m.sum()),
               "gauge_median_m3s": round(gm, 1),
               "glofas_cell_median_m3s": round(cmed, 1),
               "pearson_raw_discharge": round(r_raw, 3),
               "spearman_seasonal_pctl": round(rho, 3),
               "n_gauge_flood_days": int(flood.sum()),
               "glofas_extreme_hit_rate": (round(hit, 3)
                                           if hit is not None else None)}
        results.append(res)
        print(f"{g['name']}")
        print(f"  medians: gauge {gm:,.0f} vs cell {cmed:,.0f} m3/s   "
              f"days {res['n_days']:,}")
        print(f"  raw discharge r = {r_raw:.3f}   "
              f"seasonal-percentile rho = {rho:.3f}")
        print(f"  gauge flood days (>=98th pctl): {flood.sum():,}  ->  "
              f"GloFAS >=95th on {hit:.0%} of them\n" if hit is not None
              else "")

    out = ROOT / "models" / "runs" / "flood-validation.json"
    out.write_text(json.dumps({
        "name": "flood-validation", "layer": "flood",
        "method": ("GloFAS v5 seasonal flow percentiles vs USGS gauge "
                   "percentiles, PNW pilot"),
        "gauges": results,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
