"""Fire-weather daily series and danger features from NASA POWER.

The fire layer follows the brief exactly: model fire *danger conditions*, not
ignition. Most fires are human-started; what is predictable is how dangerous a
fire would be if one starts, which is what operational indices measure. Two
classical signals plus the percentile machinery proven on the landslide layer:

  KBDI  Keetch-Byram Drought Index (1968), the brief's named fuel-moisture
        proxy: a 0-800 running soil/duff moisture deficiency, driven by daily
        max temperature and rainfall against local mean annual precipitation.
        Label-free and computable anywhere POWER has weather -- i.e. globally.
  VPD   vapor pressure deficit (kPa) from Tmax and RH -- the modern dryness
        signal most correlated with fire activity in the literature.

Everything is also expressed as a percentile of that cell's own record
(all-time and season-matched), because D14 proved the relative-to-local-
climatology framing is where the skill lives, and it is what makes the index
comparable between Alabama and the Atacama.

POWER daily parameters: T2M_MAX (degC), RH2M (%), WS2M (m/s),
PRECTOTCORR (mm). One request per 0.5-degree cell, cached gzipped.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
FW_DIR = CACHE / "firepower"
FW_DIR.mkdir(parents=True, exist_ok=True)

GRID = 0.5
START = "2004-01-01"
END = "2024-12-31"
PARAMS = ["T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR"]
SPAN = {"start": START, "end": END, "params": PARAMS}

WINDOWS = (30, 90)

FEATURES = [
    "kbdi", "kbdi_pctl", "kbdi_pctl_seasonal",
    "vpd_kpa", "vpd_pctl_seasonal",
    "tmax_c", "tmax_pctl_seasonal",
    "rh_pct", "rh_pctl_seasonal",
    "ws_ms", "ws_pctl_seasonal",
    "days_since_rain",
    "precip_30d_pctl_seasonal", "precip_90d_pctl_seasonal",
]


def cell(lat: float, lon: float) -> tuple[float, float]:
    return (round(round(lat / GRID) * GRID, 2), round(round(lon / GRID) * GRID, 2))


def _path(clat: float, clon: float) -> Path:
    return FW_DIR / f"{clat:+06.2f}_{clon:+07.2f}.json.gz"


def fetch_cell(clat: float, clon: float) -> dict | None:
    p = _path(clat, clon)
    if p.exists():
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("_span") == SPAN:
                return d
        except (OSError, EOFError, json.JSONDecodeError):
            pass
    r = SESSION.get(API, params={
        "parameters": ",".join(PARAMS), "community": "AG",
        "latitude": clat, "longitude": clon,
        "start": START.replace("-", ""), "end": END.replace("-", ""),
        "format": "JSON"}, timeout=300)
    if r.status_code != 200:
        return None
    try:
        par = r.json()["properties"]["parameter"]
    except (KeyError, ValueError):
        return None
    keys = sorted(par.get("PRECTOTCORR", {}))
    if not keys:
        return None
    out = {"_span": SPAN,
           "time": [f"{k[:4]}-{k[4:6]}-{k[6:]}" for k in keys]}
    for pm in PARAMS:
        src = par.get(pm, {})
        out[pm] = [None if src.get(k, -999) <= -900 else float(src[k]) for k in keys]
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    return out


def fetch_cells(cells, workers: int = 6, verbose: bool = True) -> int:
    todo = [c for c in cells if not _path(*c).exists()]
    if verbose:
        print(f"    {len(cells)-len(todo)} cached, {len(todo)} to fetch", flush=True)
    got = 0
    if not todo:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_cell, *c): c for c in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                if f.result():
                    got += 1
            except Exception as e:                          # noqa: BLE001
                if verbose:
                    print(f"    {futs[f]} failed: {e}", flush=True)
            if verbose and n % 20 == 0:
                print(f"    {n}/{len(todo)}", end="\r", flush=True)
    if verbose:
        print(f"    fetched {got}/{len(todo)}        ", flush=True)
    return got


# ------------------------------------------------------------------- KBDI ---

def kbdi_series(tmax_c: np.ndarray, precip_mm: np.ndarray) -> np.ndarray:
    """Daily KBDI (0-800, hundredths of an inch of moisture deficiency).

    Keetch & Byram (1968) as operationalised by the US Forest Service:
      - only the part of a wet spell beyond 0.20 in reduces the deficiency;
      - drying is driven by Tmax and scaled by mean annual rainfall, so the
        index self-adapts to local climate (a built-in climatology term).
    Initialised at 0 (saturated); the first year is spin-up and callers using
    percentiles are insensitive to it.
    """
    t_f = np.nan_to_num(tmax_c, nan=15.0) * 9.0 / 5.0 + 32.0
    rain_in = np.nan_to_num(precip_mm, nan=0.0) / 25.4
    n = len(rain_in)
    years = max(1.0, n / 365.25)
    r_annual = float(rain_in.sum() / years)          # mean annual precip, inches

    dry_denom = 1.0 + 10.88 * math.exp(-0.0441 * r_annual)
    q = 0.0
    spell = 0.0            # cumulative rain in the current wet spell
    out = np.empty(n)
    for i in range(n):
        r = rain_in[i]
        if r > 0.0:
            prev = spell
            spell += r
            net = (spell - 0.20 if prev >= 0.20 else max(0.0, spell - 0.20))
            if prev >= 0.20:
                net = r                                # threshold already paid
            if net > 0.0:
                q = max(0.0, q - net * 100.0)
        else:
            spell = 0.0
        if t_f[i] > 50.0:
            dq = ((800.0 - q) * (0.968 * math.exp(0.0486 * t_f[i]) - 8.30)
                  * 1e-3 / dry_denom)
            q = min(800.0, q + max(0.0, dq))
        out[i] = q
    return out


class FireCellSeries:
    """Feature computation for one cell, mirroring the landslide CellSeries."""

    def __init__(self, raw: dict):
        t = raw["time"]
        self.idx = {d: i for i, d in enumerate(t)}
        g = lambda k: np.array([np.nan if v is None else v for v in raw[k]],
                               dtype="float64")
        self.tmax = g("T2M_MAX")
        self.rh = g("RH2M")
        self.ws = g("WS2M")
        self.precip = g("PRECTOTCORR")
        self.doy = np.array([int(d[5:7]) * 31 + int(d[8:10]) for d in t])
        self.kbdi = kbdi_series(self.tmax, self.precip)
        # VPD (kPa): saturation vapor pressure at Tmax x (1 - RH)
        es = 0.6108 * np.exp(17.27 * self.tmax / (self.tmax + 237.3))
        self.vpd = es * (1.0 - np.clip(self.rh, 0, 100) / 100.0)
        # days since >=2.5mm rain
        self.dsr = np.zeros(len(t))
        run = 0.0
        for i in range(len(t)):
            run = 0.0 if (self.precip[i] or 0) >= 2.5 else run + 1.0
            self.dsr[i] = run
        filled = np.nan_to_num(self.precip, nan=0.0)
        self._cum = np.concatenate([[0.0], np.cumsum(filled)])

    def _wsum(self, w: int) -> np.ndarray:
        s = self._cum[w:] - self._cum[:-w]
        return np.concatenate([np.full(w - 1, np.nan), s])

    def _pctl(self, arr: np.ndarray, i: int) -> float:
        fin = arr[np.isfinite(arr)]
        return float((fin <= arr[i]).mean()) if fin.size else float("nan")

    def _pctl_seasonal(self, arr: np.ndarray, i: int) -> float:
        m = np.isfinite(arr) & (np.abs(self.doy - self.doy[i]) <= 15)
        hs = arr[m]
        return float((hs <= arr[i]).mean()) if hs.size >= 30 else float("nan")

    def features(self, date: str) -> dict | None:
        i = self.idx.get(date)
        if i is None or i < 365:                    # KBDI spin-up year
            return None
        out = {
            "kbdi": float(self.kbdi[i]),
            "kbdi_pctl": self._pctl(self.kbdi, i),
            "kbdi_pctl_seasonal": self._pctl_seasonal(self.kbdi, i),
            "vpd_kpa": float(self.vpd[i]) if np.isfinite(self.vpd[i]) else float("nan"),
            "vpd_pctl_seasonal": self._pctl_seasonal(self.vpd, i),
            "tmax_c": float(self.tmax[i]) if np.isfinite(self.tmax[i]) else float("nan"),
            "tmax_pctl_seasonal": self._pctl_seasonal(self.tmax, i),
            "rh_pct": float(self.rh[i]) if np.isfinite(self.rh[i]) else float("nan"),
            "rh_pctl_seasonal": self._pctl_seasonal(self.rh, i),
            "ws_ms": float(self.ws[i]) if np.isfinite(self.ws[i]) else float("nan"),
            "ws_pctl_seasonal": self._pctl_seasonal(self.ws, i),
            "days_since_rain": float(self.dsr[i]),
        }
        for w in WINDOWS:
            ws = self._wsum(w)
            out[f"precip_{w}d_pctl_seasonal"] = self._pctl_seasonal(ws, i)
        return out


_CACHE: dict[tuple[float, float], FireCellSeries | None] = {}


def series_for(lat: float, lon: float) -> FireCellSeries | None:
    key = cell(lat, lon)
    if key not in _CACHE:
        raw = fetch_cell(*key)
        _CACHE[key] = FireCellSeries(raw) if raw else None
    return _CACHE[key]


def features_at(lat: float, lon: float, date: str) -> dict | None:
    s = series_for(lat, lon)
    return s.features(date) if s else None


if __name__ == "__main__":
    # sanity: a fire-season vs winter day in interior California
    for d in ("2020-08-16", "2020-02-10"):
        f = features_at(39.75, -121.5, d)          # Paradise / Camp Fire country
        print(f"\n(39.75, -121.5)  {d}")
        for k in FEATURES:
            v = f.get(k, float("nan")) if f else float("nan")
            print(f"  {k:<26} {v:9.3f}")
