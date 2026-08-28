"""Daily rainfall from NASA POWER — the unmetered trigger source.

Why this exists alongside `openmeteo.py`: Open-Meteo serves ERA5-Land at 0.1
degrees, which is the better grid, but its free tier prices a request by
locations x variables x days and a 218-cell x 21-year pull exhausts the hourly
quota (see docs/decisions.md D9). NASA POWER serves the same shape of data with
no key and no practical rate limit — one 21-year point series returns in ~1.5 s.

Tradeoff, recorded rather than hidden:

  resolution   POWER ~0.5 deg (MERRA-2) vs ERA5-Land 0.1 deg. In the Pacific
               Northwest, rainfall varies sharply over 55 km (coast / valley /
               Cascade crest), so a POWER cell smooths real orographic
               gradients and two points 30 km apart can share one series.
  bias         This costs resolution, not validity. The case-crossover design
               compares a cell against ITSELF on other dates, and every feature
               is a percentile against that cell's own record — so a coarse
               cell adds noise to the trigger signal but cannot bias it toward
               either class.
  product      POWER's PRECTOTCORR is bias-corrected against gauge climatology;
               it reads wetter than ERA5 here (2015-12-08: 87 mm vs 22 mm at
               the same point). Neither is ground truth; percentiles make the
               choice largely immaterial.

`openmeteo.py` remains the higher-resolution upgrade path — its cache is
span-aware and resumable, so it can be filled in across hourly windows later
and swapped in without touching the model code.
"""
from __future__ import annotations

import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402
from pipelines.common import SESSION  # noqa: E402
from pipelines.openmeteo import (END, START, WINDOWS, CellSeries,  # noqa: E402
                                 FEATURES)

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PW_DIR = CACHE / "nasapower"
PW_DIR.mkdir(parents=True, exist_ok=True)

GRID = 0.5                       # MERRA-2 native-ish; finer requests just resample
SPAN = {"start": START, "end": END, "source": "nasapower/PRECTOTCORR"}

__all__ = ["cell", "fetch_cells", "features_at", "series_for", "FEATURES",
           "WINDOWS", "START", "END"]


def cell(lat: float, lon: float) -> tuple[float, float]:
    return (round(round(lat / GRID) * GRID, 2), round(round(lon / GRID) * GRID, 2))


def _path(clat: float, clon: float) -> Path:
    return PW_DIR / f"{clat:+06.2f}_{clon:+07.2f}.json.gz"


def _read_cache(clat: float, clon: float) -> dict | None:
    p = _path(clat, clon)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError):
        return None
    return d if d.get("_span") == SPAN else None


def fetch_cell(clat: float, clon: float) -> dict | None:
    """One cell's daily precipitation series, cached."""
    cached = _read_cache(clat, clon)
    if cached is not None:
        return cached
    r = SESSION.get(API, params={
        "parameters": "PRECTOTCORR", "community": "AG",
        "latitude": clat, "longitude": clon,
        "start": START.replace("-", ""), "end": END.replace("-", ""),
        "format": "JSON",
    }, timeout=300)
    if r.status_code != 200:
        return None
    try:
        pr = r.json()["properties"]["parameter"]["PRECTOTCORR"]
    except (KeyError, ValueError):
        return None
    keys = sorted(pr)
    if not keys:
        return None
    out = {
        "_span": SPAN,
        "time": [f"{k[:4]}-{k[4:6]}-{k[6:]}" for k in keys],
        # POWER uses -999 for missing; map to None so CellSeries sees NaN
        "precipitation_sum": [None if pr[k] <= -900 else float(pr[k]) for k in keys],
    }
    with gzip.open(_path(clat, clon), "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def fetch_cells(cells, workers: int = 6, verbose: bool = True) -> int:
    todo = [c for c in cells if _read_cache(*c) is None]
    if verbose:
        print(f"    {len(cells)-len(todo)} cached, {len(todo)} to fetch "
              f"({workers} workers)", flush=True)
    if not todo:
        return 0
    got = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_cell, *c): c for c in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                if f.result():
                    got += 1
            except Exception as e:                          # noqa: BLE001
                if verbose:
                    print(f"    cell {futs[f]} failed: {e}", flush=True)
            if verbose:
                print(f"    cells {n}/{len(todo)}", end="\r", flush=True)
    if verbose:
        print(f"    fetched {got}/{len(todo)}          ", flush=True)
    return got


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
    for la, lo, d in [(45.5, -123.5, "2015-12-08"), (45.5, -123.5, "2015-07-15")]:
        f = features_at(la, lo, d)
        print(f"\n({la}, {lo})  {d}")
        for k in FEATURES:
            print(f"  {k:<28} {f.get(k, float('nan')):9.3f}" if f else "  no data")
