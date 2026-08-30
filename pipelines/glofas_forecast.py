"""GloFAS *forecast* discharge: gives rivers a "today" and a "this week".

The historical product (D25) ends in 2024, so a live site needs the daily
operational forecast: `cems-glofas-forecast`, system `operational`, LISFLOOD
control run, 0.05 deg, init daily, leads 24..720 h. We pull a small window
around a point for the freshest init and the first 7 leads -- a request tiny
enough that per-location daily cost is negligible.

The number served is still the platform's one move: the forecast discharge
ranked against THAT CELL's own 21-year seasonal history (the D25 stack), on
the forecast's validity date. Forecast grids and history grids are both
0.05 deg; ranking happens at the exact channel cell the historical stack
chose, so forecast and climatology describe the same river.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402

FC_DIR = CACHE / "glofas_forecast"
FC_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "cems-glofas-forecast"
LEADS = [24, 48, 72, 96, 120, 144, 168]          # one week
BOX_HALF = 0.5                                    # deg around the point


def _tag(lat: float, lon: float) -> str:
    return f"{lat:+06.2f}_{lon:+07.2f}".replace("-", "m")


def fetch(lat: float, lon: float, init: str) -> Path | None:
    """Control-forecast NetCDF for a small box around (lat, lon), one init."""
    out = FC_DIR / f"fc_{_tag(lat, lon)}_{init}.nc"
    if out.exists() and out.stat().st_size > 0:
        return out
    import cdsapi
    c = cdsapi.Client(quiet=True)
    y, m, d = init.split("-")
    try:
        c.retrieve(DATASET, {
            "system_version": ["operational"],
            "hydrological_model": ["lisflood"],
            "product_type": ["control_forecast"],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "year": [y], "month": [m], "day": [d],
            "leadtime_hour": [str(h) for h in LEADS],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [lat + BOX_HALF, lon - BOX_HALF,
                     lat - BOX_HALF, lon + BOX_HALF],
        }, str(out))
    except Exception as e:                                  # noqa: BLE001
        out.unlink(missing_ok=True)
        last = str(e).splitlines()[-1][:90] if str(e) else "?"
        print(f"    forecast {init}: {last}", flush=True)
        return None
    return out


def latest(lat: float, lon: float, max_back: int = 5):
    """Freshest available init date and its file, trying today backwards."""
    for back in range(1, max_back + 1):
        init = (dt.date.today() - dt.timedelta(days=back)).isoformat()
        p = fetch(lat, lon, init)
        if p is not None:
            return init, p
    return None, None


def outlook(lat: float, lon: float, stack) -> dict | None:
    """Ranked river outlook at the stack's channel cell near (lat, lon).

    Returns today's (first lead) and the week-max seasonal percentile, both
    ranked against the historical stack -- or None when no forecast is
    fetchable. `stack` is a pipelines.glofas.DischargeStack.
    """
    import xarray as xr
    init, path = latest(lat, lon)
    if path is None:
        return None
    ds = xr.open_dataset(path)
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    da = ds[var]

    # the channel cell the HISTORY chose is the river we rank against
    probe = stack.percentile_at(lat, lon, stack.dates[-1])
    if not probe or not probe.get("is_river"):
        ds.close()
        return {"is_river": False}
    lat_name = next(d for d in da.dims if "lat" in d.lower())
    lon_name = next(d for d in da.dims if "lon" in d.lower())
    s_lat = next(d for d in stack.da.dims if "lat" in d.lower())
    s_lon = next(d for d in stack.da.dims if "lon" in d.lower())
    # locate history's chosen channel cell coordinates via its own snap
    hist_cell = None
    best_med = -1
    for dla in (-0.10, -0.05, 0.0, 0.05, 0.10):
        for dlo in (-0.10, -0.05, 0.0, 0.05, 0.10):
            rv = stack.river.sel({s_lat: lat + dla,
                                  s_lon: stack.q_lon(lon) + dlo},
                                 method="nearest")
            if not bool(rv):
                continue
            cc = stack.da.sel({s_lat: lat + dla,
                               s_lon: stack.q_lon(lon) + dlo},
                              method="nearest")
            med = float(np.nanmedian(cc.values))
            if med > best_med:
                best_med = med
                hist_cell = (float(cc[s_lat]), float(cc[s_lon]))
    if hist_cell is None:
        ds.close()
        return {"is_river": False}
    h_lat, h_lon = hist_cell
    f_lon = h_lon if float(da[lon_name].max()) > 180 else (
        h_lon - 360.0 if h_lon > 180 else h_lon)

    # rank each lead's value on the history's seasonal window for its own
    # validity date
    series = stack.da.sel({s_lat: h_lat, s_lon: h_lon}, method="nearest").values
    init_d = dt.date.fromisoformat(init)
    leads_out = []
    for k, lead in enumerate(LEADS):
        try:
            val = float(da.isel({d: 0 for d in da.dims
                                 if d not in (lat_name, lon_name,
                                              "forecast_period",
                                              "leadtime_hour", "step")})
                        .isel({next(d for d in da.dims if d in
                                    ("forecast_period", "leadtime_hour",
                                     "step")): k})
                        .sel({lat_name: h_lat, lon_name: f_lon},
                             method="nearest").values)
        except Exception:                                   # noqa: BLE001
            continue
        if not np.isfinite(val):
            continue
        vdate = init_d + dt.timedelta(hours=lead)
        key = vdate.month * 31 + vdate.day
        m = np.abs(stack.doy_key - key) <= 15
        hist = series[m]
        hist = hist[np.isfinite(hist)]
        if hist.size < 30:
            continue
        leads_out.append({
            "valid": vdate.isoformat(), "lead_h": lead,
            "discharge_m3s": round(val, 1),
            "flow_pctl_seasonal": round(float((hist <= val).mean()), 3)})
    ds.close()
    if not leads_out:
        return None
    week_max = max(leads_out, key=lambda x: x["flow_pctl_seasonal"])
    return {"is_river": True, "init": init,
            "today": leads_out[0], "week_max": week_max, "leads": leads_out,
            "median_here_m3s": round(best_med, 1)}


if __name__ == "__main__":
    from pipelines.glofas import stack_for
    stack, why = stack_for(23.8, 90.4)          # Dhaka
    if stack is None:
        raise SystemExit(why)
    r = outlook(23.8, 90.4, stack)
    import json
    print(json.dumps({k: v for k, v in (r or {}).items() if k != "leads"},
                     indent=1))
    for L in (r or {}).get("leads", []):
        print(f"  +{L['lead_h']:>3}h  {L['valid']}  "
              f"{L['discharge_m3s']:>9,.0f} m3/s  pctl {L['flow_pctl_seasonal']:.3f}")
