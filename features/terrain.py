"""Terrain derivatives from Copernicus DEM GLO-30.

These are the SUSCEPTIBILITY-side features: slow-changing, intrinsic properties
of a location. Everything here is computed from a local elevation patch, so one
tile read yields the whole set.

Metric spacing matters. A 1-arcsec pixel is ~30.9 m north-south everywhere, but
only 30.9*cos(lat) m east-west -- about 21.7 m at 45N. Computing slope on raw
degree spacing (a common bug) inflates east-west gradients by ~1/cos(lat) and
biases aspect toward east-west. We convert to metres before differencing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.dem import ensure_tile, pixel_metres, tile_name  # noqa: E402

HALF = 16  # patch half-width in pixels -> 33x33 ~ 1 km at 45N

FEATURES = [
    "elev_m", "slope_deg", "aspect_sin", "aspect_cos", "curvature",
    "tri", "roughness_std", "relief_range", "tpi",
]

_OPEN: dict[str, rasterio.DatasetReader] = {}


def _dataset(name: str):
    if name in _OPEN:
        return _OPEN[name]
    p = ensure_tile(name)
    if p is None:
        _OPEN[name] = None
        return None
    _OPEN[name] = rasterio.open(p)
    return _OPEN[name]


def patch(lat: float, lon: float, half: int = HALF):
    """Elevation patch centred on (lat, lon). Returns None off-tile/ocean.

    Patches that would cross a tile boundary are clipped rather than mosaicked;
    the caller sees a smaller array. Affects only points within ~500 m of a
    whole-degree line.
    """
    ds = _dataset(tile_name(lat, lon))
    if ds is None:
        return None
    row, col = ds.index(lon, lat)
    r0, c0 = row - half, col - half
    r1, c1 = row + half + 1, col + half + 1
    r0c, c0c = max(0, r0), max(0, c0)
    r1c, c1c = min(ds.height, r1), min(ds.width, c1)
    if r1c - r0c < 5 or c1c - c0c < 5:
        return None
    a = ds.read(1, window=Window(c0c, r0c, c1c - c0c, r1c - r0c)).astype("float64")
    nod = ds.nodata
    if nod is not None:
        a[a == nod] = np.nan
    a[a < -400] = np.nan          # GLO-30 sentinel / bad values
    if np.isnan(a).mean() > 0.5:
        return None
    return a


def derive(lat: float, lon: float, half: int = HALF) -> dict | None:
    """All terrain features at a point, or None where the DEM has no data."""
    a = patch(lat, lon, half)
    if a is None:
        return None
    dx, dy = pixel_metres(lat)
    ci, cj = a.shape[0] // 2, a.shape[1] // 2
    centre = a[ci, cj]
    if not np.isfinite(centre):
        # fall back to the patch median so a single bad pixel is not fatal
        centre = float(np.nanmedian(a))
        if not np.isfinite(centre):
            return None

    # fill holes with the patch median so gradients stay defined
    filled = np.where(np.isfinite(a), a, np.nanmedian(a))

    # Horn slope/aspect on the 3x3 around the centre
    if filled.shape[0] >= 3 and filled.shape[1] >= 3:
        w = filled[ci - 1:ci + 2, cj - 1:cj + 2]
        if w.shape != (3, 3):
            w = filled[max(0, ci - 1):ci + 2, max(0, cj - 1):cj + 2]
    dzdx = dzdy = 0.0
    if w.shape == (3, 3):
        dzdx = ((w[0, 2] + 2 * w[1, 2] + w[2, 2]) -
                (w[0, 0] + 2 * w[1, 0] + w[2, 0])) / (8 * dx)
        dzdy = ((w[2, 0] + 2 * w[2, 1] + w[2, 2]) -
                (w[0, 0] + 2 * w[0, 1] + w[0, 2])) / (8 * dy)
    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = math.atan2(dzdy, -dzdx)

    # curvature: discrete Laplacian at the centre, per 100 m
    curv = 0.0
    if w.shape == (3, 3):
        curv = ((w[1, 0] - 2 * w[1, 1] + w[1, 2]) / (dx * dx) +
                (w[0, 1] - 2 * w[1, 1] + w[2, 1]) / (dy * dy)) * 1e4

    # terrain ruggedness index: mean |difference| to the 8 neighbours
    tri = float(np.mean(np.abs(w - w[1, 1]))) if w.shape == (3, 3) else float("nan")

    return {
        "elev_m": float(centre),
        "slope_deg": float(slope),
        "aspect_sin": float(math.sin(aspect)),
        "aspect_cos": float(math.cos(aspect)),
        "curvature": float(curv),
        "tri": tri,
        "roughness_std": float(np.nanstd(a)),
        "relief_range": float(np.nanmax(a) - np.nanmin(a)),
        "tpi": float(centre - np.nanmean(a)),
    }


def close_all() -> None:
    for ds in _OPEN.values():
        if ds is not None:
            ds.close()
    _OPEN.clear()


if __name__ == "__main__":
    for la, lo, what in [(45.52, -122.68, "Portland OR (flat, urban)"),
                         (45.37, -121.70, "Mt Hood flank (steep)"),
                         (46.85, -121.76, "Mt Rainier (very steep)")]:
        d = derive(la, lo)
        print(f"\n{what}  ({la}, {lo})")
        if d is None:
            print("  no DEM data")
        else:
            for k in FEATURES:
                print(f"  {k:<15} {d[k]:10.3f}")
    close_all()
