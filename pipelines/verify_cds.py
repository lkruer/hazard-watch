"""One-shot check that the EWDS/GloFAS account is wired up.

Run after creating the account and pasting the Personal Access Token into
C:/Users/<you>/.cdsapirc (url + key lines). Makes the smallest sensible
request -- one day of GloFAS river discharge over the PNW box, NetCDF -- and
prints what came back. Requests queue server-side, so first success can take
a few minutes; that is normal.

Failure translations:
  401 / "not authenticated"      token missing or pasted wrong in .cdsapirc
  403 / "required licences"      the CEMS-FLOODS licence box was not accepted
                                 on the dataset's Download tab (one-time click)
  404 dataset                    wrong portal -- GloFAS lives on
                                 ewds.climate.copernicus.eu, not cds.*
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402

OUT = CACHE / "glofas"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import cdsapi

    target = OUT / "verify_pnw_20171122.nc"
    c = cdsapi.Client()
    print("requesting 1 day of GloFAS discharge over the PNW box "
          "(queued server-side; a few minutes is normal)...")
    # field names/values verified against the live process schema 2026-08-28
    base = {
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["average_river_discharge_in_the_last_24_hours"],
        "year": ["2017"],
        "month": ["11"],
        "day": ["22"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [49.2, -124.8, 42.0, -120.5],   # N, W, S, E
    }
    last_err = None
    for sv, ts in (("version_5_0", "time_mean"), ("version_5_0", "instantaneous"),
                   ("version_4_0", "time_mean"), ("version_4_0", "instantaneous")):
        try:
            c.retrieve("cems-glofas-historical",
                       {**base, "system_version": [sv], "timespan": [ts]},
                       str(target))
            print(f"combination accepted: {sv} / {ts}")
            last_err = None
            break
        except Exception as e:                              # noqa: BLE001
            if "valid combination" in str(e):
                last_err = e
                continue
            raise
    if last_err is not None:
        raise last_err
    print(f"downloaded {target.stat().st_size/1e6:.1f} MB -> {target.name}")

    import xarray as xr
    ds = xr.open_dataset(target)
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    d = ds[var]
    print(f"variable {var}: shape {tuple(d.shape)}")
    print(f"discharge over box: max {float(d.max()):.0f} m3/s, "
          f"mean {float(d.mean()):.1f} m3/s")
    print("\nCDS/EWDS access VERIFIED — the flood layer is unblocked.")


if __name__ == "__main__":
    main()
