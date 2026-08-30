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


def _fort_shift(f: int, k: int) -> int:
    return int((f + k) % N_FORT)


def _fortnight_of(date: str) -> int:
    d = dt.date.fromisoformat(date)
    return min((d.timetuple().tm_yday - 1) // 14, N_FORT - 1)


def _rank(value: np.ndarray, ladder: np.ndarray) -> np.ndarray:
    """Percentile of value within a (21, lat, lon) quantile ladder,
    LINEARLY INTERPOLATED between rungs.

    Counting rungs alone quantizes to steps of 1/21 ~ 0.048, so the highest
    rank below "ties the all-time max" is 0.952 -- which silently made the
    >=0.98 rain-alert gate unreachable (Sindh read 0.95 at the peak of the
    2022 Pakistan floods and did not flag). Interpolation restores continuous
    percentiles and the gate's intended meaning."""
    k = np.sum(ladder <= value[None, ...], axis=0)          # rungs at/below v
    kc = np.clip(k, 1, len(QS) - 1)
    idx_lo = (kc - 1)[None, ...]
    lo = np.take_along_axis(ladder, idx_lo, axis=0)[0]
    hi = np.take_along_axis(ladder, idx_lo + 1, axis=0)[0]
    span = hi - lo
    frac = np.where(span > 0, (value - lo) / np.where(span > 0, span, 1.0), 1.0)
    q_step = 1.0 / (len(QS) - 1)
    out = (QS[kc - 1] + np.clip(frac, 0.0, 1.0) * q_step).astype("float32")
    out = np.where(k == 0, 0.0, out)
    out = np.where(k >= len(QS), 1.0, out)
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

    cache = PROCESSED / "world" / f"window_{date}.npz"
    if cache.exists():
        w = np.load(cache)
        P, TX, RH, WS = w["P"], w["TX"], w["RH"], w["WS"]
        print(f"trailing window from cache ({cache.name})")
    else:
        ds = xr.open_zarr(ZARR, consolidated=True,
                          storage_options={"anon": True})
        t1 = np.datetime64(date)
        t0 = t1 - np.timedelta64(400, "D")
        win = ds.sel(time=slice(str(t0), date))
        print(f"pulling trailing window {str(t0)[:10]}..{date} from S3 ...")
        P = win["PRECTOTCORR"].values.astype("float32")
        TX = win["T2M_MAX"].values.astype("float32")
        RH = win["RH2M"].values.astype("float32")
        WS = win["WS2M"].values.astype("float32")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, P=P, TX=TX, RH=RH, WS=WS)   # uncompressed: the 1.3GB compressed write failed silently
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
        # seed the short-window recursion at each cell's climatological
        # median KBDI for the window-START fortnight (ladder q-index 10)
        # climatological mean-annual precip per cell from the spi365 ladder
        # median, averaged over fortnights -- already on disk, no rebuild
        "kbdi": _kbdi_band(
            TX, P,
            q0=z["kbdi"][_fort_shift(f, -(P.shape[0] // 14)), 10],
            r_annual_mm=np.nanmean(z["spi365"][:, 10], axis=0))[-1],
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

    # exposure denominator (D28): people per POWER cell, GHS-POP CC-BY 4.0
    pop_path = PROCESSED / "population_power_grid.npy"
    pop = np.load(pop_path) if pop_path.exists() else None

    summary = {"date": date, "grid": [int(len(lat)), int(len(lon))]}
    for k, m in alerts.items():
        n = int(np.nansum(m))
        summary[k] = {"cells": n}
        if pop is not None:
            summary[k]["people"] = int(np.nansum(np.where(m, pop, 0.0)))
        if n:
            sig = {"landslide_rain": "rain3d", "fire": "vpd",
                   "drought": "spi90"}[k]
            # highlight the most extreme POPULATED cell -- POWER covers ocean,
            # and an empty Southern Ocean cell is not a useful example
            keep = m if pop is None else (m & (pop > 1000))
            if not keep.any():
                keep = m
            fld = np.where(keep, pct[sig], np.nan)
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

    # snapped onto the zarr grid so world and point pipelines sample the SAME
    # physical cell -- off-grid points differ by up to ~35 km, which across a
    # Sierra or Himalayan gradient is a different climate (parity noise, not
    # method error; flat Iowa agreed to +/-0.03 even unsnapped)
    def snap(la, lo):
        return (float(lat[np.argmin(np.abs(lat - la))]),
                float(lon[np.argmin(np.abs(lon - lo))]))
    POINTS = [(n, *snap(la, lo)) for n, la, lo in
              [("pnw-coast-range", 45.5, -123.5),
               ("kathmandu", 27.75, 85.40),
               ("paradise-CA", 39.75, -121.60),
               ("okavango", -19.5, 23.0),
               ("iowa", 41.9, -93.6)]]
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
