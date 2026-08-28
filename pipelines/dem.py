"""Copernicus DEM GLO-30 access.

Public AWS S3 bucket, no credentials, Cloud-Optimized GeoTIFF, 1-arcsec (~30 m).
Tiles are 1 deg x 1 deg, named by their SOUTH-WEST corner:

    Copernicus_DSM_COG_10_N44_00_W124_00_DEM/Copernicus_DSM_COG_10_N44_00_W124_00_DEM.tif

Verified live 2026-08. The brief names either SRTM (USGS/SRTMGL1_003) or
Copernicus GLO-30; GLO-30 is chosen because it is reachable without Earth
Engine authentication, so the terrain layer is not blocked on EE onboarding.

Tiles are downloaded once and cached under data/cache/dem/. A tile is ~40 MB;
the Pacific Northwest study box needs roughly 20 of them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_DIR = CACHE / "dem"
DEM_DIR.mkdir(parents=True, exist_ok=True)


def tile_name(lat: float, lon: float) -> str:
    """SW-corner tile name containing (lat, lon)."""
    la, lo = math.floor(lat), math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"


def tile_path(name: str) -> Path:
    return DEM_DIR / f"{name}.tif"


def ensure_tile(name: str) -> Path | None:
    """Download a tile if absent. Returns None if the tile does not exist
    (ocean tiles are simply missing from the bucket)."""
    p = tile_path(name)
    if p.exists() and p.stat().st_size > 0:
        return p
    marker = DEM_DIR / f"{name}.missing"
    if marker.exists():
        return None

    url = f"{BUCKET}/{name}/{name}.tif"
    r = SESSION.get(url, stream=True, timeout=600)
    if r.status_code == 404:
        marker.write_text("404", encoding="utf-8")
        return None
    r.raise_for_status()

    tmp = p.with_suffix(".part")
    total = int(r.headers.get("Content-Length") or 0)
    got = 0
    with tmp.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            got += len(chunk)
            if total:
                print(f"    {name}  {got/1e6:6.1f}/{total/1e6:.1f} MB", end="\r", flush=True)
    tmp.replace(p)
    print(f"    {name}  {got/1e6:6.1f} MB cached           ")
    return p


def tiles_for_points(points) -> set[str]:
    """Distinct tile names covering an iterable of (lat, lon)."""
    return {tile_name(la, lo) for la, lo in points}


def pixel_metres(lat: float) -> tuple[float, float]:
    """Ground size of a 1-arcsec pixel at this latitude: (dx, dy) in metres."""
    dy = 111_320.0 / 3600.0
    dx = dy * math.cos(math.radians(lat))
    return dx, dy


if __name__ == "__main__":
    # smoke test
    n = tile_name(45.3, -123.7)
    print("tile for (45.3, -123.7):", n)
    p = ensure_tile(n)
    print("cached at:", p)
    if p:
        import rasterio
        with rasterio.open(p) as ds:
            print("  size:", ds.width, "x", ds.height, " crs:", ds.crs,
                  " nodata:", ds.nodata, " dtype:", ds.dtypes[0])
