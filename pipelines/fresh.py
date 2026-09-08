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
ERA5_API = "https://archive-api.open-meteo.com/v1/archive"

# POWER's normal publication lag is 3-4 days. Beyond this, the board goes
# stale for a reason upstream of us (observed 2026-09: POWER stuck 9 days
# behind while ERA5T was same-day), and we bridge the gap from ERA5 --
# bias-corrected against the pair's own overlap, per combine_verdict's law
# that constant bias is harmless to percentiles but story changes are not.
GAPFILL_AFTER_DAYS = 5

# POWER field -> (Open-Meteo daily field, correction mode). "ratio" absorbs
# multiplicative bias and unit/height differences (WS2M is m/s at 2 m,
# windspeed_10m_mean km/h at 10 m -- the overlap ratio eats both); "offset"
# fits additive bias for temperature-like fields.
ERA5_MAP = {
    "PRECTOTCORR": ("precipitation_sum", "ratio"),
    "precipitation_sum": ("precipitation_sum", "ratio"),
    "T2M_MAX": ("temperature_2m_max", "offset"),
    "RH2M": ("relative_humidity_2m_mean", "offset"),
    "WS2M": ("windspeed_10m_mean", "ratio"),
}
ERA5_DAILY = ("precipitation_sum,temperature_2m_max,"
              "relative_humidity_2m_mean,windspeed_10m_mean")
MIN_OVERLAP_DAYS = 120


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


def _era5_tail(clat: float, clon: float) -> dict | None:
    """Recent daily ERA5/ERA5T series (all 4 vars), cached with the same TTL.

    One small request per cell per night; ERA5T runs to ~today, so this is
    the freshest public daily surface we have when POWER stalls.
    """
    p = _tail_path("era5", clat, clon)
    if _tail_fresh_enough(p):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    try:
        r = SESSION.get(ERA5_API, params={
            "latitude": clat, "longitude": clon,
            "start_date": TAIL_START,
            "end_date": dt.date.today().isoformat(),
            "daily": ERA5_DAILY, "timezone": "UTC"}, timeout=180)
    except Exception:                                       # noqa: BLE001
        r = None
    if r is None or r.status_code != 200:
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        return None
    d = r.json().get("daily") or {}
    if not d.get("time"):
        return None
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(d, fh)
    return d


def _gapfill(merged: dict, clat: float, clon: float,
             fields: list[str]) -> dict | None:
    """Fill each field's TRAILING gap from bias-corrected ERA5, in place.

    Only trailing missing days (after the field's last finite value) are
    touched -- mid-history gaps are real absences and stay absent, so the
    climatology the percentiles rank against is unchanged. A field is filled
    only when its lag exceeds GAPFILL_AFTER_DAYS; in normal operation the
    series stays pure POWER. Returns an info dict when anything was filled.
    """
    era5 = None
    today = dt.date.today()
    idx = {d: i for i, d in enumerate(merged["time"])}
    info: dict = {"source": "ERA5 (Open-Meteo)", "fields": {}}
    for field in fields:
        spec = ERA5_MAP.get(field)
        if spec is None:
            continue
        e_name, mode = spec
        vals = merged.get(field)
        if not vals:
            continue
        last = max((i for i, v in enumerate(vals) if v is not None),
                   default=None)
        if last is None:
            continue
        lag = (today - dt.date.fromisoformat(merged["time"][last])).days
        if lag <= GAPFILL_AFTER_DAYS:
            continue
        if era5 is None:
            era5 = _era5_tail(clat, clon)
            if era5 is None:
                return None
            e_idx = {d: i for i, d in enumerate(era5["time"])}
        ev = era5.get(e_name)
        if not ev:
            continue
        # correction from the pair's own overlap (shared finite days)
        pv_o, ev_o = [], []
        for d_, i in idx.items():
            j = e_idx.get(d_)
            if j is None or vals[i] is None or ev[j] is None:
                continue
            pv_o.append(vals[i]); ev_o.append(ev[j])
        if len(pv_o) < MIN_OVERLAP_DAYS:
            continue
        if mode == "ratio":
            corr = (sum(pv_o) + 1.0) / (sum(ev_o) + 1.0)
            # wind's ratio legitimately sits near 0.2 (km/h at 10 m -> m/s
            # at 2 m); precip's sanity band is tighter
            lo = 0.05 if field == "WS2M" else 0.25
            corr = min(4.0, max(lo, corr))
        else:
            corr = sum(p - e for p, e in zip(pv_o, ev_o)) / len(pv_o)
            corr = min(10.0, max(-10.0, corr))
        n = 0
        last_date = merged["time"][last]
        for d_ in sorted(e_idx):
            if d_ <= last_date or e_idx[d_] is None:
                continue
            j = e_idx[d_]
            if ev[j] is None:
                continue
            v = ev[j] * corr if mode == "ratio" else ev[j] + corr
            if field == "RH2M":
                v = min(100.0, max(1.0, v))
            i = idx.get(d_)
            if i is None:            # ERA5 reaches past the POWER time axis
                merged["time"].append(d_)
                for f2 in fields:
                    if f2 in merged:
                        merged[f2].append(None)
                i = len(merged["time"]) - 1
                idx[d_] = i
            merged[field][i] = round(float(v), 3)
            n += 1
        if n:
            info["fields"][field] = {
                "days": n, "mode": mode, "correction": round(corr, 3),
                "power_through": last_date}
    return info if info["fields"] else None


def rain_series(lat: float, lon: float,
                info_out: dict | None = None) -> CellSeries | None:
    """CellSeries (rain trigger + SPI machinery) through ~3 days ago.

    When POWER's tail stalls past its normal lag, trailing days come from
    bias-corrected ERA5 and `info_out` (if given) records what was filled.
    """
    c = nasapower.cell(lat, lon)
    hist = nasapower.fetch_cell(*c)
    tail = _fetch_tail(["PRECTOTCORR"], *c, kind="rain")
    if hist is None or tail is None:
        return CellSeries(hist) if hist else None
    merged = _merge(hist, tail, {"precipitation_sum": "PRECTOTCORR"})
    gi = _gapfill(merged, *c, fields=["precipitation_sum"])
    if gi and info_out is not None:
        info_out.update(gi)
    return CellSeries(merged)


def fire_series(lat: float, lon: float,
                info_out: dict | None = None) -> fw.FireCellSeries | None:
    """FireCellSeries through the freshest common date of the 4 fire vars."""
    c = fw.cell(lat, lon)
    hist = fw.fetch_cell(*c)
    tail = _fetch_tail(fw.PARAMS, *c, kind="fire")
    if hist is None or tail is None:
        return fw.FireCellSeries(hist) if hist else None
    merged = _merge(hist, tail, {p: p for p in fw.PARAMS})
    gi = _gapfill(merged, *c, fields=list(fw.PARAMS))
    if gi and info_out is not None:
        info_out.update(gi)
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
