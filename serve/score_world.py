"""Score the ENTIRE world for one date: global percentile fields per hazard.

Consumes the climatology ladders built by pipelines/power_global.py plus a
single pull of the trailing global weather from the same NASA POWER Zarr
store. Rank each cell's value on its own fortnight ladder -> the identical
seasonal-percentile semantics every layer validated with, now as world maps.

Outputs (data/processed/world/<date>/):
  <signal>_pctl.npy      float16 percentile field (361 x 576), NaN off-record
  alerts.npz             boolean masks per hazard at deployment thresholds
  summary.json           counts + the day's most extreme cells per hazard

Runtime after ladders exist: one ~400 MB S3 read (trailing 365 days of
precip + the day's fire fields), then seconds of numpy.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED  # noqa: E402
from pipelines.power_global import (N_FORT, QS, ZARR, _kbdi_band,  # noqa: E402
                                    _rolling_sum)

LADDERS = PROCESSED / "global_ladders.npz"

ALERT = {
    "landslide_rain": ("rain3d", 0.98),      # with rain30d >= .95 as OR-gate
    "fire": ("vpd", 0.95),                   # vpd AND kbdi extreme, see below
    "drought": ("spi90", 0.05),              # low tail
}


def _fortnight_of(date: str) -> int:
    d = dt.date.fromisoformat(date)
    return min((d.timetuple().tm_yday - 1) // 14, N_FORT - 1)


def _rank(value: np.ndarray, ladder: np.ndarray) -> np.ndarray:
    """Percentile of value within a (21, lat, lon) quantile ladder."""
    v = value[None, ...]
    below = np.sum(ladder <= v, axis=0).astype("float32")
    out = below / float(len(QS))
    out[~np.isfinite(value)] = np.nan
    out[~np.isfinite(ladder).all(axis=0)] = np.nan
    return out


def main(date: str) -> None:
    import xarray as xr
    z = np.load(LADDERS)
    lat, lon = z["lat"], z["lon"]
    f = _fortnight_of(date)
    print(f"ladders loaded (clim {z['clim_start']}..{z['clim_end']}), "
          f"fortnight {f}")

    ds = xr.open_zarr(ZARR, consolidated=True, storage_options={"anon": True})
    t1 = np.datetime64(date)
    t0 = t1 - np.timedelta64(400, "D")
    win = ds.sel(time=slice(str(t0), date))
    print(f"pulling trailing window {str(t0)[:10]}..{date} from S3 ...")
    P = win["PRECTOTCORR"].values.astype("float32")
    TX = win["T2M_MAX"].values.astype("float32")
    RH = win["RH2M"].values.astype("float32")
    WS = win["WS2M"].values.astype("float32")
    if not np.isfinite(P[-1]).any():
        raise SystemExit(f"{date} beyond the archive's real data")

    es = 0.6108 * np.exp(17.27 * TX / (TX + 237.3))
    vpd = es * (1.0 - np.clip(RH, 0, 100) / 100.0)
    today = {
        "rain3d": _rolling_sum(P, 3)[-1],
        "rain30d": _rolling_sum(P, 30)[-1],
        "spi90": _rolling_sum(P, 90)[-1],
        "spi180": _rolling_sum(P, 180)[-1],
        "spi365": _rolling_sum(P, 365)[-1],
        "kbdi": _kbdi_band(TX, P)[-1],
        "vpd": vpd[-1],
        "tmax": TX[-1],
        "ws": WS[-1],
    }

    out_dir = PROCESSED / "world" / date
    out_dir.mkdir(parents=True, exist_ok=True)
    pct = {}
    for s, v in today.items():
        pct[s] = _rank(v, z[s][f])
        np.save(out_dir / f"{s}_pctl.npy", pct[s].astype("float16"))

    alerts = {
        "landslide_rain": (pct["rain3d"] >= 0.98) | (pct["rain30d"] >= 0.98),
        "fire": (pct["vpd"] >= 0.95) & (pct["kbdi"] >= 0.90),
        "drought": pct["spi90"] <= 0.05,
    }
    np.savez_compressed(out_dir / "alerts.npz", **alerts)

    summary = {"date": date, "grid": [int(len(lat)), int(len(lon))]}
    for k, m in alerts.items():
        n = int(np.nansum(m))
        summary[k] = {"cells": n}
        if n:
            sig = {"landslide_rain": "rain3d", "fire": "vpd",
                   "drought": "spi90"}[k]
            fld = np.where(m, pct[sig], np.nan)
            idx = np.unravel_index(np.nanargmax(
                fld if k != "drought" else -fld), fld.shape)
            summary[k]["most_extreme"] = {
                "lat": float(lat[idx[0]]), "lon": float(lon[idx[1]]),
                "pctl": float(pct[sig][idx])}
        print(f"  {k:<15} {n:5d} cells flagged"
              + (f"   e.g. {summary[k].get('most_extreme')}" if n else ""))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"wrote {out_dir}")


def parity(date: str) -> None:
    """Compare world-field percentiles against the validated point pipelines.

    Exact equality is not expected: the world ladders use fortnight bins and a
    21-step quantile ladder over 2001+, while the point pipelines use a
    +/-15-day sliding window over 2004+. Agreement within ~0.10 mean absolute
    difference is the pass bar; larger drift would mean the world maps say
    something different from what was validated, which would be disqualifying.
    """
    from pipelines import fireweather as fw
    from pipelines import nasapower
    from eval.drought_validate import DroughtSeries

    z = np.load(LADDERS)
    lat, lon = z["lat"], z["lon"]
    src = PROCESSED / "world" / date
    fields = {s: np.load(src / f"{s}_pctl.npy").astype("float32")
              for s in ("rain3d", "rain30d", "spi90", "kbdi", "vpd")}

    POINTS = [("pnw-coast-range", 45.5, -123.5),
              ("kathmandu", 27.75, 85.40),
              ("paradise-CA", 39.75, -121.60),
              ("okavango", -19.5, 23.0),
              ("iowa", 41.9, -93.6)]
    diffs = []
    print(f"{'point':<18}{'signal':<10}{'world':>7}{'point':>7}{'diff':>7}")
    for name, la, lo in POINTS:
        i = int(np.argmin(np.abs(lat - la)))
        j = int(np.argmin(np.abs(lon - lo)))
        w = nasapower.features_at(la, lo, date) or {}
        fwf = fw.features_at(la, lo, date) or {}
        raw = nasapower.fetch_cell(*nasapower.cell(la, lo))
        dsf = DroughtSeries(raw).features(date) if raw else {}
        ref = {"rain3d": w.get("precip_3d_pctl_seasonal"),
               "rain30d": w.get("precip_30d_pctl_seasonal"),
               "spi90": (dsf or {}).get("spi90"),
               "kbdi": fwf.get("kbdi_pctl_seasonal"),
               "vpd": fwf.get("vpd_pctl_seasonal")}
        for s, rv in ref.items():
            wv = float(fields[s][i, j])
            if rv is None or not np.isfinite(wv):
                continue
            d = wv - rv
            diffs.append(abs(d))
            print(f"{name:<18}{s:<10}{wv:>7.2f}{rv:>7.2f}{d:>+7.2f}")
    mad = float(np.mean(diffs))
    print(f"\nmean |diff| = {mad:.3f} over {len(diffs)} comparisons  "
          f"-> {'PASS (<=0.10)' if mad <= 0.10 else 'INVESTIGATE'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-11-15")
    ap.add_argument("--parity", action="store_true",
                    help="compare world fields to the point pipelines")
    a = ap.parse_args()
    main(a.date)
    if a.parity:
        parity(a.date)
