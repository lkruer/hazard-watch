"""Fresh weather tails: bring point series to within days of now.

The bulk zarr archive lags ~a month; the POWER point API lags 3-4 days
(measured 2026-08-30: PRECTOTCORR to 08-27, T2M_MAX to 08-26) and ERA5 via
Open-Meteo is same-day. A daily product therefore scores from the POINT API,
same MERRA-2 family as every ladder and validation, refreshed nightly.

Design: the long historical caches (2004-2024) stay immutable; this module
fetches only a RECENT TAIL (2025-01-01 -> today) per cell, cached with a 20h
TTL, and concatenates. Nightly cost per registered cell: one or two small API
calls. Percentiles computed on the merged series keep the exact semantics the
hindcasts validated -- the seasonal window simply gains the newest year.

D31's law applies with force here: the freshest data is where archives fail
first, so update_locations verifies the tail against ERA5 before serving
anything derived from it.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402
from pipelines.common import SESSION  # noqa: E402
from pipelines import fireweather as fw  # noqa: E402
from pipelines import nasapower  # noqa: E402
from pipelines.openmeteo import CellSeries  # noqa: E402

FRESH_DIR = CACHE / "fresh"
FRESH_DIR.mkdir(parents=True, exist_ok=True)
TAIL_START = "2025-01-01"          # historical caches end 2024-12-31
TTL_HOURS = 20
API = "https://power.larc.nasa.gov/api/temporal/daily/point"


def _tail_path(kind: str, clat: float, clon: float) -> Path:
    return FRESH_DIR / f"{kind}_{clat:+06.2f}_{clon:+07.2f}.json.gz"


def _tail_fresh_enough(p: Path) -> bool:
    return (p.exists()
            and (time.time() - p.stat().st_mtime) < TTL_HOURS * 3600)


def _fetch_tail(params: list[str], clat: float, clon: float,
                kind: str) -> dict | None:
    p = _tail_path(kind, clat, clon)
    if _tail_fresh_enough(p):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    r = SESSION.get(API, params={
        "parameters": ",".join(params), "community": "AG",
        "latitude": clat, "longitude": clon,
        "start": TAIL_START.replace("-", ""),
        "end": dt.date.today().strftime("%Y%m%d"),
        "format": "JSON"}, timeout=180)
    if r.status_code != 200:
        # stale tail beats no tail; say nothing here, caller sees the dates
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        return None
    par = r.json()["properties"]["parameter"]
    keys = sorted(par[params[0]])
    out = {"time": [f"{k[:4]}-{k[4:6]}-{k[6:]}" for k in keys]}
    for pm in params:
        src = par.get(pm, {})
        out[pm] = [None if src.get(k, -999) <= -900 else float(src[k])
                   for k in keys]
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def _merge(hist: dict, tail: dict, fields: dict[str, str]) -> dict:
    """Historical series + recent tail -> one series dict.

    fields maps output-name -> tail-name (historical dicts already use the
    output names). History wins before TAIL_START; tail wins after.
    """
    idx = {}
    for i, d in enumerate(hist["time"]):
        if d < TAIL_START:
            idx[d] = ("h", i)
    for i, d in enumerate(tail["time"]):
        idx[d] = ("t", i)
    days = sorted(idx)
    out = {"time": days}
    for out_name, tail_name in fields.items():
        hv, tv = hist.get(out_name, []), tail.get(tail_name, [])
        col = []
        for d in days:
            src, i = idx[d]
            col.append(hv[i] if src == "h" and i < len(hv)
                       else (tv[i] if src == "t" and i < len(tv) else None))
        out[out_name] = col
    return out


def rain_series(lat: float, lon: float) -> CellSeries | None:
    """CellSeries (rain trigger + SPI machinery) through ~3 days ago."""
    c = nasapower.cell(lat, lon)
    hist = nasapower.fetch_cell(*c)
    tail = _fetch_tail(["PRECTOTCORR"], *c, kind="rain")
    if hist is None or tail is None:
        return CellSeries(hist) if hist else None
    return CellSeries(_merge(hist, tail,
                             {"precipitation_sum": "PRECTOTCORR"}))


def fire_series(lat: float, lon: float) -> fw.FireCellSeries | None:
    """FireCellSeries through the freshest common date of the 4 fire vars."""
    c = fw.cell(lat, lon)
    hist = fw.fetch_cell(*c)
    tail = _fetch_tail(fw.PARAMS, *c, kind="fire")
    if hist is None or tail is None:
        return fw.FireCellSeries(hist) if hist else None
    merged = _merge(hist, tail, {p: p for p in fw.PARAMS})
    return fw.FireCellSeries(merged)


def last_valid_date(series, attr: str = "precip") -> str | None:
    import numpy as np
    v = getattr(series, attr)
    ok = np.where(np.isfinite(v))[0]
    if not ok.size:
        return None
    days = list(series.idx)
    return days[int(ok[-1])]


def power_tail_total(lat: float, lon: float, days: int = 120) -> float | None:
    """Recent POWER precip total straight from the tail -- feed to
    precip_quality.recent_vs_era5 (D31: verify the freshest window)."""
    import numpy as np
    s = rain_series(lat, lon)
    if s is None:
        return None
    return float(np.nansum(s.precip[-days:]))


if __name__ == "__main__":
    s = rain_series(45.5, -123.5)
    print("rain series through:", last_valid_date(s))
    f = fire_series(45.5, -123.5)
    print("fire series through:", last_valid_date(f, "tmax"))
    d = last_valid_date(s)
    print("features on freshest day:",
          {k: round(v, 3) for k, v in list(s.features(d).items())[:4]})
