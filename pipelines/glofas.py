"""GloFAS river discharge: fetch, cache, and flow percentiles.

Per the brief, flood is an INTEGRATION task, not a modeling task: GloFAS
(Copernicus CEMS, EWDS) already runs the hydrology. What we add is the same
move as every other layer -- express today's discharge as a percentile of
that river cell's own multi-year record for this time of year. "High water"
means something different on the Mekong than on a Cascades creek, exactly as
rainfall did.

Data: cems-glofas-historical, system version 5.0, LISFLOOD, daily mean
discharge, 0.05 deg grid. Fetched one calendar year per request for a bbox
(requests queue server-side), cached as NetCDF under data/cache/glofas/.

River mask: the grid covers every land cell, but a flow percentile is only
meaningful where a river actually runs. A cell qualifies if its long-record
median discharge >= MIN_RIVER_M3S; elsewhere the flood layer reports
not-applicable rather than a percentile of a dry ditch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402

GL_DIR = CACHE / "glofas"
GL_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "cems-glofas-historical"
VERSION = "version_5_0"
YEARS = list(range(2004, 2025))          # matches the platform's weather span
MIN_RIVER_M3S = 5.0


def _tag(bbox) -> str:
    n, w, s, e = bbox
    return f"{n:.1f}_{w:.1f}_{s:.1f}_{e:.1f}".replace("-", "m")


def year_path(bbox, year: int) -> Path:
    return GL_DIR / f"dis_{_tag(bbox)}_{year}.nc"


def fetch_year(bbox, year: int) -> Path | None:
    """One calendar year of daily mean discharge for a bbox (N, W, S, E)."""
    out = year_path(bbox, year)
    if out.exists() and out.stat().st_size > 0:
        return out
    import cdsapi
    c = cdsapi.Client(quiet=True)
    # ONE product type per request: asking for consolidated+intermediate
    # together doubles the request cost and trips the EWDS cap. Consolidated
    # covers the archive; only the most recent months need intermediate, and
    # the except-branch below retries with it when consolidated has no data.
    try:
        c.retrieve(DATASET, {
            "system_version": [VERSION],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["average_river_discharge_in_the_last_24_hours"],
            "timespan": ["time_mean"],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": list(bbox),
        }, str(out))
    except Exception as e:                                  # noqa: BLE001
        if year >= 2024:            # recent months live in 'intermediate'
            try:
                c.retrieve(DATASET, {
                    "system_version": [VERSION],
                    "hydrological_model": ["lisflood"],
                    "product_type": ["intermediate"],
                    "variable": ["average_river_discharge_in_the_last_24_hours"],
                    "timespan": ["time_mean"],
                    "year": [str(year)],
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    "area": list(bbox),
                }, str(out))
                return out
            except Exception as e2:                         # noqa: BLE001
                e = e2
        print(f"  [{year}] failed: {str(e).splitlines()[-1][:90]}", flush=True)
        out.unlink(missing_ok=True)
        return None
    return out


def fetch_span(bbox, years=YEARS) -> list[Path]:
    got = []
    for y in years:
        p = fetch_year(bbox, y)
        if p:
            got.append(p)
            print(f"  [{y}] ok ({p.stat().st_size/1e6:.1f} MB)", flush=True)
    return got


class DischargeStack:
    """Multi-year daily discharge for a bbox, with percentile queries."""

    def __init__(self, bbox):
        import xarray as xr
        paths = [year_path(bbox, y) for y in YEARS]
        paths = [p for p in paths if p.exists()]
        if not paths:
            raise FileNotFoundError(f"no GloFAS cache for bbox {bbox}")
        # plain concat: 21 small files, no dask dependency needed
        parts = []
        var = None
        for pth in paths:
            ds = xr.open_dataset(pth)
            if var is None:
                var = next(v for v in ds.data_vars if "dis" in v.lower())
            parts.append(ds[var].load())
            ds.close()
        tname = next(d for d in parts[0].dims if "time" in d.lower())
        self.da = xr.concat(parts, dim=tname).sortby(tname)
        tdim = next(d for d in self.da.dims if "time" in d.lower())
        self.tdim = tdim
        t = self.da[tdim].dt
        self.doy_key = (t.month * 31 + t.day).values
        self.dates = np.array([str(v)[:10] for v in self.da[tdim].values])
        med = self.da.median(dim=tdim)
        self.river = med >= MIN_RIVER_M3S
        # EWDS serves longitude on 0..360; queries arrive as -180..180
        lon_name = next(d for d in self.da.dims if "lon" in d.lower())
        self._lon360 = bool(float(self.da[lon_name].max()) > 180.0)

    def q_lon(self, lon: float) -> float:
        return lon % 360.0 if self._lon360 else lon

    def percentile_at(self, lat: float, lon: float, date: str,
                      window: int = 15) -> dict | None:
        """Seasonal flow percentile at the nearest river CHANNEL.

        A 0.05-deg channel rarely sits exactly under a queried coordinate, so
        snap to the largest-median river cell within +/-2 cells (~10 km) --
        the river a person at this point lives near. Beyond that radius,
        honestly report not-a-river."""
        lat_name = next(d for d in self.da.dims if "lat" in d.lower())
        lon_name = next(d for d in self.da.dims if "lon" in d.lower())
        qlon = self.q_lon(lon)
        best, best_med = None, -1.0
        for dla in (-0.10, -0.05, 0.0, 0.05, 0.10):
            for dlo in (-0.10, -0.05, 0.0, 0.05, 0.10):
                rv = self.river.sel({lat_name: lat + dla, lon_name: qlon + dlo},
                                    method="nearest")
                if not bool(rv):
                    continue
                cc = self.da.sel({lat_name: lat + dla, lon_name: qlon + dlo},
                                 method="nearest")
                med = float(np.nanmedian(cc.values))
                if med > best_med:
                    best_med, best = med, cc
        if best is None:
            return {"is_river": False}
        cell = best
        idx = np.where(self.dates == date)[0]
        if not idx.size:
            return None
        i = int(idx[0])
        series = cell.values
        val = float(series[i])
        key = self.doy_key[i]
        m = np.abs(self.doy_key - key) <= window
        hist = series[m]
        hist = hist[np.isfinite(hist)]
        if hist.size < 30:
            return None
        return {"is_river": True,
                "discharge_m3s": round(val, 1),
                "flow_pctl_seasonal": round(float((hist <= val).mean()), 3),
                "flow_pctl_alltime": round(
                    float((series[np.isfinite(series)] <= val).mean()), 3),
                "median_here_m3s": round(float(np.median(hist)), 1)}


PNW_BBOX = (49.2, -124.8, 42.0, -120.5)          # N, W, S, E


# ------------------------------------------------- global tile-on-demand ---
# GloFAS is global but a planet of 0.05-deg daily discharge is not fetchable
# wholesale. Coverage grows the way DEM coverage does: 6x6-degree tiles,
# aligned to a fixed grid, fetched once on demand and cached forever. A new
# tile costs ~21 queued requests (roughly an hour); after that, every point
# in that basin scores instantly.

TILE_DEG = 6.0


def tile_bbox(lat: float, lon: float):
    """The fixed-grid tile (N, W, S, E) containing a point."""
    import math
    s = math.floor(lat / TILE_DEG) * TILE_DEG
    w = math.floor(lon / TILE_DEG) * TILE_DEG
    return (s + TILE_DEG, w, s, w + TILE_DEG)


def tile_years_cached(bbox) -> int:
    return sum(1 for y in YEARS if year_path(bbox, y).exists())


_STACKS: dict = {}


def stack_for(lat: float, lon: float, min_years: int = 15):
    """DischargeStack covering the point, or None with a reason.

    Prefers the PNW pilot box where it applies (deeper record first), else the
    point's fixed tile if enough years are cached. Never fetches inline --
    scoring must not block for an hour; fetch tiles with:
        python pipelines/glofas.py --tile <lat> <lon>
    """
    n, w, s, e = PNW_BBOX
    for bbox in ((PNW_BBOX,) if (s <= lat <= n and w <= lon <= e) else ()) +                 (tile_bbox(lat, lon),):
        have = tile_years_cached(bbox)
        if have >= min_years:
            key = _tag(bbox)
            if key not in _STACKS:
                _STACKS[key] = DischargeStack(bbox)
            return _STACKS[key], None
    have = tile_years_cached(tile_bbox(lat, lon))
    return None, (f"no discharge record cached for this basin yet "
                  f"({have}/{len(YEARS)} years; fetch with "
                  f"pipelines/glofas.py --tile)")


# Priority basins for progressive global coverage, ordered by flood exposure
# (population living in the floodplain). Each entry is a point inside the
# basin; its 6-degree tile gets fetched. Resumable: cached years are skipped.
PRIORITY_BASINS = [
    ("ganges-brahmaputra (Dhaka)", 23.8, 90.4),
    ("mekong delta (Can Tho)", 10.0, 105.8),
    ("indus (Sukkur)", 27.7, 68.9),
    ("yangtze (Wuhan)", 30.6, 114.3),
    ("irrawaddy (Yangon)", 17.0, 95.2),
    ("niger inland delta (Niamey)", 13.5, 2.1),
    ("nile (Khartoum)", 15.6, 32.5),
    ("mississippi (Memphis)", 35.1, -90.1),
    ("rhine-meuse (Cologne)", 50.9, 6.96),
    ("chao phraya (Bangkok)", 13.8, 100.5),
]


def fetch_priority():
    for name, la, lo in PRIORITY_BASINS:
        bb = tile_bbox(la, lo)
        have = tile_years_cached(bb)
        if have >= len(YEARS):
            print(f"== {name}: complete ({have}/{len(YEARS)})", flush=True)
            continue
        print(f"== {name}: tile {bb} ({have}/{len(YEARS)} cached)", flush=True)
        fetch_span(bb)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="fetch the 6-degree tile containing this point")
    ap.add_argument("--priority", action="store_true",
                    help="work through PRIORITY_BASINS (resumable)")
    a = ap.parse_args()
    if a.priority:
        fetch_priority()
        raise SystemExit(0)
    if a.tile:
        bb = tile_bbox(*a.tile)
        print(f"fetching {len(YEARS)} years for tile {bb}")
        fetch_span(bb)
        raise SystemExit(0)
    print(f"fetching {len(YEARS)} years of GloFAS discharge for the PNW box")
    fetch_span(PNW_BBOX)
    st = DischargeStack(PNW_BBOX)
    n_river = int(st.river.sum())
    print(f"\nriver cells (median >= {MIN_RIVER_M3S} m3/s): {n_river:,}")
    # 2017-11-22: a real high-water day in the probe download
    for name, la, lo in [("columbia-at-portland", 45.60, -122.75),
                         ("willamette-eugene", 44.05, -123.10),
                         ("skagit-mount-vernon", 48.42, -122.33)]:
        r = st.percentile_at(la, lo, "2017-11-22")
        print(f"  {name:<22} {r}")
