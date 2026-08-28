"""Rainfall + temperature history from the Open-Meteo ERA5 archive.

Free, no API key, no registration. These are the TRIGGER-side inputs.

Source choice: the brief names CHIRPS or GPM IMERG. Both are reachable but
awkward -- CHIRPS ships one global GeoTIFF per day (thousands of files for a
multi-decade record) and IMERG sits behind Earthdata Login. Open-Meteo serves
ERA5/ERA5-Land as point time series over plain HTTP, which is the same
reanalysis the brief already nominates for the fire hazard. Tradeoff worth
recording: ERA5 underestimates short convective extremes relative to the
gauge-blended CHIRPS, so absolute mm totals run low in convective regimes. We
mitigate by never using an absolute threshold -- every trigger feature is a
percentile against that cell's own record, which is what LHASA does and what
the brief requires.

Requests are deduplicated onto a 0.1 deg grid (ERA5-Land's native resolution),
so hundreds of nearby label points collapse to a handful of fetches. Each cell
is cached gzipped under data/cache/openmeteo/.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

API = "https://archive-api.open-meteo.com/v1/archive"
OM_DIR = CACHE / "openmeteo"
OM_DIR.mkdir(parents=True, exist_ok=True)

GRID = 0.1
# 21 years is a sound climatology baseline for percentile estimation, and the
# study region has only 24 labelled events before 2005 (out of 1,294) -- so the
# extra decade cost a third of the quota to buy 2% more labels.
START = "2004-01-01"
END = "2024-12-31"
# Precipitation only. Open-Meteo prices a request by locations x variables x
# days, and a 417-cell x 30-year pull with three variables exhausted the free
# hourly quota outright. Rainfall is the dominant landslide trigger; temperature
# (as a snowmelt proxy) is a real but secondary driver, and is deferred rather
# than paid for at 3x the quota cost. Adding it back is this list plus a refetch
# -- the cache is span-aware, so stale entries invalidate themselves.
DAILY = ["precipitation_sum"]

WINDOWS = (1, 3, 7, 14, 30)

FEATURES = (
    [f"precip_{w}d" for w in WINDOWS]
    + [f"precip_{w}d_pctl" for w in WINDOWS]
    + [f"precip_{w}d_pctl_seasonal" for w in WINDOWS]
    + ["precip_30d_over_climo_mean"]
)


def cell(lat: float, lon: float) -> tuple[float, float]:
    """Snap to the 0.1 deg fetch grid."""
    return (round(round(lat / GRID) * GRID, 1), round(round(lon / GRID) * GRID, 1))


def _path(clat: float, clon: float) -> Path:
    return OM_DIR / f"{clat:+06.1f}_{clon:+07.1f}.json.gz"


# Cached payloads record the span they were fetched for. Widening START/END or
# changing DAILY must not silently keep serving the old, narrower series.
SPAN = {"start": START, "end": END, "daily": list(DAILY)}


def _read_cache(clat: float, clon: float) -> dict | None:
    p = _path(clat, clon)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError):
        return None
    return d if d.get("_span") == SPAN else None       # stale -> refetch


def _write_cache(clat: float, clon: float, daily: dict) -> dict:
    out = {"_span": SPAN, "time": daily["time"],
           **{k: daily.get(k) for k in DAILY}}
    with gzip.open(_path(clat, clon), "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def fetch_cells(cells, batch: int = 10, pause: float = 1.5, verbose: bool = True,
                max_stalls: int = 4) -> int:
    """Fetch many grid cells using Open-Meteo's multi-location form.

    The archive endpoint accepts comma-separated latitude/longitude and returns
    one object per location **in request order** -- results are matched by
    position, not by the coordinates echoed back, since the API snaps each
    request to the nearest ERA5 grid node and returns the snapped value.

    One request per cell took ~4 s each; 20 per request is roughly 20x cheaper
    in wall-clock for the same data.
    """
    todo = [c for c in cells if _read_cache(*c) is None]
    if verbose:
        print(f"    {len(cells)-len(todo)} cached, {len(todo)} to fetch "
              f"in ~{(len(todo)+batch-1)//batch} batches", flush=True)
    got = 0
    i = 0
    stalls = 0
    while i < len(todo):
        chunk = todo[i:i + batch]
        try:
            r = SESSION.get(API, params={
                "latitude": ",".join(f"{c[0]}" for c in chunk),
                "longitude": ",".join(f"{c[1]}" for c in chunk),
                "start_date": START, "end_date": END,
                "daily": ",".join(DAILY), "timezone": "UTC",
            }, timeout=600)
            # Open-Meteo prices a call by locations x variables x days, so a
            # wide batch over a 30-year span burns quota fast. Back off and
            # retry the SAME chunk rather than shrinking it -- the limit is on
            # volume per unit time, so waiting is what actually helps.
            if r.status_code == 429:
                stalls += 1
                if stalls > max_stalls:
                    if verbose:
                        print(f"    still limited after {max_stalls} waits; "
                              f"stopping with {len(todo)-i} cells unfetched "
                              f"(cache is resumable -- just re-run)", flush=True)
                    break
                body = (r.text or "").lower()
                if "hourly" in body:
                    # Quota resets on the wall clock hour, so short backoff is
                    # pointless -- sleep to just past the top of the next hour.
                    now = dt.datetime.now()
                    nxt = (now.replace(minute=0, second=0, microsecond=0)
                           + dt.timedelta(hours=1, seconds=30))
                    wait = max(30.0, (nxt - now).total_seconds())
                    label = f"hourly quota; sleeping {wait/60:.1f} min until {nxt:%H:%M}"
                else:
                    wait = float(r.headers.get("Retry-After") or min(60 * stalls, 300))
                    label = f"rate limited; waiting {wait:.0f}s"
                if verbose:
                    print(f"    {label}  ({i}/{len(todo)} done)", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            stalls = 0
        except Exception as e:                              # noqa: BLE001
            if len(chunk) > 1:                              # split and retry
                if verbose:
                    print(f"    batch of {len(chunk)} failed ({e}); halving", flush=True)
                batch = max(1, len(chunk) // 2)
                continue
            if verbose:
                print(f"    cell {chunk[0]} failed permanently: {e}", flush=True)
            i += 1
            continue

        if isinstance(payload, dict):
            payload = [payload]
        for c, loc in zip(chunk, payload):
            d = (loc or {}).get("daily")
            if d and d.get("time"):
                _write_cache(*c, d)
                got += 1
        i += len(chunk)
        if verbose:
            print(f"    cells {min(i, len(todo))}/{len(todo)}", end="\r", flush=True)
        time.sleep(pause)          # be a good citizen on a free endpoint
    if verbose and todo:
        print()
    return got


def fetch_cell(clat: float, clon: float) -> dict | None:
    """Daily series for one grid cell, cached. None if the API has no data."""
    d = _read_cache(clat, clon)
    if d is not None:
        return d
    fetch_cells([(clat, clon)], batch=1, verbose=False)
    return _read_cache(clat, clon)


class CellSeries:
    """Indexed daily series for one cell, with climatology helpers."""

    __slots__ = ("idx", "precip", "doy", "_cum")

    def __init__(self, raw: dict):
        t = raw["time"]
        self.idx = {d: i for i, d in enumerate(t)}
        self.precip = np.array([np.nan if v is None else v
                                for v in raw["precipitation_sum"]], dtype="float64")
        self.doy = np.array([int(d[5:7]) * 31 + int(d[8:10]) for d in t])
        filled = np.nan_to_num(self.precip, nan=0.0)
        self._cum = np.concatenate([[0.0], np.cumsum(filled)])

    def window_sums(self, w: int) -> np.ndarray:
        """Rolling w-day totals ending at each index (NaN for the first w-1)."""
        s = self._cum[w:] - self._cum[:-w]
        return np.concatenate([np.full(w - 1, np.nan), s])

    def features(self, date: str) -> dict | None:
        i = self.idx.get(date)
        if i is None or i < max(WINDOWS):
            return None
        out: dict[str, float] = {}
        for w in WINDOWS:
            ws = self.window_sums(w)
            val = float(ws[i])
            out[f"precip_{w}d"] = val

            hist = ws[np.isfinite(ws)]
            out[f"precip_{w}d_pctl"] = (
                float((hist <= val).mean()) if hist.size else float("nan"))

            # seasonal: same time of year (+/- 15 "day units") across all years
            target = self.doy[i]
            m = np.isfinite(ws) & (np.abs(self.doy - target) <= 15)
            hs = ws[m]
            out[f"precip_{w}d_pctl_seasonal"] = (
                float((hs <= val).mean()) if hs.size >= 30 else float("nan"))

        ws30 = self.window_sums(30)
        mean30 = np.nanmean(ws30)
        out["precip_30d_over_climo_mean"] = (
            float(ws30[i] / mean30) if mean30 and np.isfinite(mean30) and mean30 > 0
            else float("nan"))
        return out


_CACHE: dict[tuple[float, float], CellSeries | None] = {}


def series_for(lat: float, lon: float) -> CellSeries | None:
    key = cell(lat, lon)
    if key not in _CACHE:
        raw = fetch_cell(*key)
        _CACHE[key] = CellSeries(raw) if raw else None
    return _CACHE[key]


def features_at(lat: float, lon: float, date: str) -> dict | None:
    s = series_for(lat, lon)
    return s.features(date) if s else None


if __name__ == "__main__":
    # Oregon Coast Range, a real wet-season day
    for la, lo, dt_ in [(45.5, -123.5, "2015-12-08"), (45.5, -123.5, "2015-07-15")]:
        f = features_at(la, lo, dt_)
        print(f"\n({la}, {lo})  {dt_}")
        if not f:
            print("  no data")
        else:
            for k in FEATURES:
                v = f.get(k, float("nan"))
                print(f"  {k:<28} {v:9.3f}")
