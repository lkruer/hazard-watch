"""Global climatology ladders from the NASA POWER Zarr archive on AWS.

The step change for whole-world coverage: NASA publishes the ENTIRE POWER
daily record (MERRA-2, 0.5 x 0.625 deg, 1981-present) as an anonymously
readable Zarr store on S3 -- the same data the point API serves, so every
validation done against the API carries over unchanged.

This module streams it once, in latitude bands, and reduces 25+ years of
daily history to what the scorer actually needs: for each cell, each signal,
and each fortnight-of-year, a 21-step quantile ladder. Rank today's value on
the ladder -> the same seasonal percentile the point pipelines compute, for
every cell on Earth at once.

Signals (matching the validated per-hazard features):
  rain3d, rain30d            landslide trigger windows
  spi90, spi180, spi365      drought windows
  kbdi, vpd, tmax, ws        fire-weather (KBDI via its sequential recursion)

Output: data/processed/global_ladders.npz  (~1-2 GB, float32)
  arrays [signal] of shape (26 fortnights, 21 quantiles, 361 lat, 576 lon)
Resumable per band: data/cache/power_global/band_<i>.npz

Cost: one pass over ~5 vars x 26 years x global grid, ~30-35 GB S3 transfer.
Bands are independent; interrupt and re-run freely.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE, PROCESSED  # noqa: E402

ZARR = "s3://nasa-power/merra2/temporal/power_merra2_daily_temporal_lst.zarr"
BAND_DIR = CACHE / "power_global"
BAND_DIR.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"          # 365-day windows spin up through 2000
CLIM_START = "2001-01-01"     # ladders use 2001+
QS = np.linspace(0.0, 1.0, 21).astype("float32")
N_FORT = 26
BAND_ROWS = 20
SIGNALS = ["rain3d", "rain30d", "spi90", "spi180", "spi365",
           "kbdi", "vpd", "tmax", "ws"]


def _open():
    import xarray as xr
    return xr.open_zarr(ZARR, consolidated=True,
                        storage_options={"anon": True})


def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing w-day sum along axis 0, NaN for the first w-1 days.

    prefix[i] = sum of the first i days; out[i] = prefix[i+1] - prefix[i+1-w]
    -- the same prefix-sum form as CellSeries._cum, which parity-checked
    against the validated point features."""
    prefix = np.concatenate(
        [np.zeros((1,) + a.shape[1:], dtype="float64"),
         np.cumsum(np.nan_to_num(a, nan=0.0), axis=0, dtype="float64")], axis=0)
    out = np.full_like(a, np.nan, dtype="float32")
    out[w - 1:] = (prefix[w:] - prefix[:-w]).astype("float32")
    return out


def _kbdi_band(tmax_c: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """Vectorized KBDI over a (time, rows, cols) band -- same recursion as
    pipelines/fireweather.kbdi_series, applied to all cells at once."""
    t_f = np.nan_to_num(tmax_c, nan=15.0) * 9.0 / 5.0 + 32.0
    rain_in = np.nan_to_num(precip, nan=0.0) / 25.4
    n = rain_in.shape[0]
    years = max(1.0, n / 365.25)
    r_annual = rain_in.sum(axis=0) / years
    dry_denom = 1.0 + 10.88 * np.exp(-0.0441 * r_annual)

    q = np.zeros(rain_in.shape[1:], dtype="float64")
    spell = np.zeros_like(q)
    out = np.empty_like(rain_in, dtype="float32")
    for i in range(n):
        r = rain_in[i]
        wet = r > 0.0
        prev = spell.copy()
        spell = np.where(wet, spell + r, 0.0)
        net = np.where(wet & (prev >= 0.20), r,
                       np.where(wet, np.maximum(0.0, spell - 0.20), 0.0))
        q = np.maximum(0.0, q - net * 100.0)
        hot = t_f[i] > 50.0
        dq = ((800.0 - q) * (0.968 * np.exp(0.0486 * t_f[i]) - 8.30)
              * 1e-3 / dry_denom)
        q = np.where(hot, np.minimum(800.0, q + np.maximum(0.0, dq)), q)
        out[i] = q
    return out


def _fortnight(times) -> np.ndarray:
    import pandas as pd
    t = pd.DatetimeIndex(times)
    return np.minimum((t.dayofyear - 1) // 14, N_FORT - 1).to_numpy()


def build_band(ds, i0: int, i1: int, times, fort, clim_mask) -> dict:
    """Quantile ladders for lat rows [i0:i1)."""
    sel = dict(time=slice(None), lat=slice(i0, i1))
    print(f"  reading vars for rows {i0}..{i1} ...", flush=True)
    P = ds["PRECTOTCORR"].isel(**sel).values.astype("float32")
    TX = ds["T2M_MAX"].isel(**sel).values.astype("float32")
    RH = ds["RH2M"].isel(**sel).values.astype("float32")
    WS = ds["WS2M"].isel(**sel).values.astype("float32")

    es = 0.6108 * np.exp(17.27 * TX / (TX + 237.3))
    vpd = (es * (1.0 - np.clip(RH, 0, 100) / 100.0)).astype("float32")

    series = {
        "rain3d": _rolling_sum(P, 3),
        "rain30d": _rolling_sum(P, 30),
        "spi90": _rolling_sum(P, 90),
        "spi180": _rolling_sum(P, 180),
        "spi365": _rolling_sum(P, 365),
        "kbdi": _kbdi_band(TX, P),
        "vpd": vpd,
        "tmax": TX,
        "ws": WS,
    }
    out = {}
    nrow = i1 - i0
    for name, arr in series.items():
        lad = np.full((N_FORT, len(QS), nrow, arr.shape[2]), np.nan,
                      dtype="float32")
        for f in range(N_FORT):
            m = clim_mask & (fort == f)
            sub = arr[m]
            if not sub.shape[0]:
                continue
            lad[f] = np.nanquantile(sub, QS, axis=0).astype("float32")
        out[name] = lad
    return out


def build(start_row: int = 0) -> None:
    ds = _open()
    ds = ds.sel(time=slice(START, None))
    # the store is padded years into the future -- cut at last real precip day
    p_last = ds["PRECTOTCORR"].isel(lat=180, lon=288).values
    real = np.where(np.isfinite(p_last))[0]
    ds = ds.isel(time=slice(0, int(real[-1]) + 1))
    times = ds.time.values
    fort = _fortnight(times)
    clim_mask = times >= np.datetime64(CLIM_START)
    nlat = ds.sizes["lat"]
    print(f"time {str(times[0])[:10]}..{str(times[-1])[:10]}  "
          f"({len(times):,} days), {nlat} lat rows", flush=True)

    for i0 in range(start_row, nlat, BAND_ROWS):
        i1 = min(i0 + BAND_ROWS, nlat)
        bp = BAND_DIR / f"band_{i0:03d}.npz"
        if bp.exists():
            print(f"band {i0:03d}: cached", flush=True)
            continue
        t0 = time.time()
        out = build_band(ds, i0, i1, times, fort, clim_mask)
        np.savez_compressed(bp, **out)
        print(f"band {i0:03d}: done in {time.time()-t0:.0f}s "
              f"({bp.stat().st_size/1e6:.0f} MB)", flush=True)

    # assemble
    print("assembling global ladders...", flush=True)
    full = {s: np.full((N_FORT, len(QS), nlat, ds.sizes["lon"]), np.nan,
                       dtype="float32") for s in SIGNALS}
    for i0 in range(0, nlat, BAND_ROWS):
        bp = BAND_DIR / f"band_{i0:03d}.npz"
        if not bp.exists():
            print(f"  band {i0:03d} missing -- skipped rows stay NaN")
            continue
        z = np.load(bp)
        i1 = min(i0 + BAND_ROWS, nlat)
        for s in SIGNALS:
            full[s][:, :, i0:i1, :] = z[s]
    meta = {"lat": ds.lat.values.astype("float32"),
            "lon": ds.lon.values.astype("float32"),
            "quantiles": QS,
            "clim_start": CLIM_START,
            "clim_end": str(times[-1])[:10]}
    out_path = PROCESSED / "global_ladders.npz"
    np.savez_compressed(out_path, **full, **meta)
    print(f"wrote {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-row", type=int, default=0)
    a = ap.parse_args()
    build(a.start_row)
