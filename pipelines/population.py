"""Population exposure: how many people a hazard cell actually contains.

A percentile field says a place is extreme; it cannot say whether that
matters. "417 cells flagged" is a number an operator cannot act on;
"this alert covers 2.3M people" is. This module supplies the missing
denominator for every layer the platform already scores.

Source: GHS-POP R2023A (EU JRC Global Human Settlement Layer), the global
30 arcsec (~1 km) WGS84 raster. Chosen over WorldPop on three grounds:
the WorldPop 1km mosaic path returned 404 while the JRC host served a
clean 482 MB in one pass; GHSL ships in EPSG:4326 so it aligns to the
POWER lat/lon grid with no reprojection step to get wrong; and its cells
carry population COUNTS, so aggregating by SUM is exact -- no area
weighting, no latitude correction, no resampling kernel to defend.

Two consumers, one raster:

  people_near(lat, lon, radius_km)   windowed read around a point, for the
                                     per-location reports serve/score.py
                                     writes. Never loads the full raster.
  build_power_grid()                 the whole world summed onto the NASA
                                     POWER grid (361 lat x 576 lon), so
                                     score_world.py can multiply an alert
                                     mask by people and report exposure.

Three properties of the real file drove the aggregation, none of them the
idealised geometry one would assume (all measured, not guessed):

  1. It is 43202 x 21384, not 43200 x 21600, and its origin is
     -180.00792 deg -- a ~0.0037 deg phase offset from the POWER grid. So
     POWER cell edges fall MID-pixel. Pixels are therefore assigned by
     centre: every pixel lands in exactly one cell, which keeps the global
     total exact and confines the offset to a sub-pixel (~460 m) wobble in
     how two adjacent cells split a boundary.
  2. It spans 360.0167 deg -- two columns MORE than the globe. Columns 0-1
     and 43200-43201 cover the same ground (they are 360.000 deg apart) and
     disagree, holding 284 and 1,456 people respectively. Summing all
     43202 columns would double-count that strip, so the build uses a
     contiguous 43200-column window covering the globe exactly once.
  3. It stops at +/-89.1 deg, so POWER rows 0-1 and 359-360 are empty by
     construction. GHS-POP maps no one above 89.1 deg; that is a property
     of the source, not a gap in this code.

POWER's first longitude cell is centred ON the antimeridian and so wraps,
taking pixels from both edges of the raster; _lon_index() handles that with
a modulo. Where the column runs happen to fold evenly the build uses a fast
roll-and-reshape, but it proves that fold against a brute-force per-pixel
bincount first and silently falls back if it disagrees.

Outputs:
  data/processed/population_power_grid.npy    float32 (361, 576), people
  data/processed/population_power_grid.json   source/licence/epoch/total

Cost: one 482 MB download (once, into data/raw/population/), then ~1 min of
streaming reads to build the grid. Nothing here touches the network again.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, RAW  # noqa: E402

POP_DIR = RAW / "population"
RASTER = POP_DIR / "GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"
GRID_NPY = PROCESSED / "population_power_grid.npy"
GRID_JSON = PROCESSED / "population_power_grid.json"
LADDERS = PROCESSED / "global_ladders.npz"

SOURCE = {
    "dataset": "GHS-POP R2023A (Global Human Settlement Layer)",
    "product": "GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0",
    "epoch": 2025,
    "resolution": "30 arcsec (~1 km at the equator)",
    "crs": "EPSG:4326",
    "producer": "European Commission Joint Research Centre (JRC)",
    "licence": "CC BY 4.0",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": (
        "Schiavina M., Freire S., Carioli A., MacManus K. (2023): "
        "GHS-POP R2023A - GHS population grid multitemporal (1975-2030). "
        "European Commission, Joint Research Centre (JRC). "
        "doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE"
    ),
    "url": (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
        "GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_4326_30ss/V1-0/"
        "GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.zip"
    ),
    "redistribution": (
        "CC BY 4.0 permits redistribution of derived aggregates provided "
        "credit is given and changes are indicated; the POWER-grid sum is "
        "such a derivative, and the change is stated in 'aggregation'."
    ),
    "licence_verified": (
        "copyright.txt fetched from the product directory and kept at "
        "data/raw/population/GHSL_copyright.txt: '(c) European Union, "
        "1995-2026 ... licensed under the Creative Commons Attribution 4.0 "
        "International (CC BY 4.0) licence. Reuse is allowed provided "
        "appropriate credit is given and any changes are indicated.'"
    ),
    "caveat": (
        "Epoch 2025 is GHSL's projected epoch, disaggregated from the 2020 "
        "observation base. Used because the platform scores today and "
        "exposure should answer 'who is there now'."
    ),
    "known_limitations": (
        "Coverage stops at +/-89.1 deg, so POWER rows 0-1 and 359-360 are "
        "empty. Extreme-latitude settlements are under-mapped: a 2-deg box "
        "around Longyearbyen, Svalbard (~2,900 residents) holds a single "
        "4.3-person pixel, so people_near there returns ~0. Mid-latitude "
        "towns are accurate -- Tromso, Norway reads 77,751 against ~77,000 "
        "actual. Treat sub-Arctic zero-population reads as unmapped, not "
        "as evidence nobody is there."
    ),
}

# NASA POWER / MERRA-2 grid, matching pipelines/power_global.py.
N_LAT, N_LON = 361, 576
DLAT, DLON = 0.5, 0.625


def power_axes() -> tuple[np.ndarray, np.ndarray]:
    """The POWER grid coordinates, preferring the ladders' own axes.

    power_global.py writes lat/lon straight from the Zarr store, so once the
    ladders exist they are the authority; before that the grid is the
    documented MERRA-2 geometry. The two are asserted equal when both exist,
    which is how a silent half-cell offset gets caught instead of shipped.
    """
    lat = -90.0 + DLAT * np.arange(N_LAT)
    lon = -180.0 + DLON * np.arange(N_LON)
    if LADDERS.exists():
        try:
            z = np.load(LADDERS)
            zlat, zlon = z["lat"].astype("float64"), z["lon"].astype("float64")
            if zlat.shape == lat.shape and zlon.shape == lon.shape:
                if (np.abs(zlat - lat).max() < 1e-3
                        and np.abs(zlon - lon).max() < 1e-3):
                    return zlat, zlon
                raise SystemExit(
                    "global_ladders.npz axes disagree with the documented "
                    "POWER grid by more than 1e-3 deg -- refusing to guess")
        except (OSError, KeyError):
            pass          # ladders mid-write; the constructed grid is correct
    return lat, lon


def _globe_cols(src) -> int:
    """Column count covering 360 deg exactly once.

    GHS-POP's global mosaic overshoots the globe by two columns whose ground
    is already covered at the far edge (measured: the pairs sit 360.000 deg
    apart and hold 284 vs 1,456 people, i.e. they disagree). Reading a
    contiguous 360-deg window instead of the whole width counts every place
    once; the choice moves ~1.2k people, 1.5e-7 of the global total.
    """
    return min(src.width, int(round(360.0 / src.transform.a)))


def _open(path: Path | None = None):
    import rasterio
    p = path or RASTER
    if not p.exists():
        raise SystemExit(
            f"missing {p}\nRun: python pipelines/population.py --download")
    return rasterio.open(p)


# ---------------------------------------------------------------- point query

def _haversine_km(lat0: float, lon0: float,
                  lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from (lat0, lon0) to each (lat, lon)."""
    r0, r1 = math.radians(lat0), np.radians(lat)
    dlat = r1 - r0
    dlon = np.radians(((lon - lon0 + 180.0) % 360.0) - 180.0)
    a = (np.sin(dlat / 2.0) ** 2
         + math.cos(r0) * np.cos(r1) * np.sin(dlon / 2.0) ** 2)
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def people_near(lat: float, lon: float, radius_km: float = 10.0,
                src=None) -> int:
    """Population within radius_km of a point, by windowed read.

    Counts a source pixel when its CENTRE falls inside the circle -- at 1 km
    pixels against a 10 km default radius the boundary error is well under a
    percent and unbiased, and the alternative (fractional pixel areas) would
    imply a within-pixel distribution GHS-POP does not claim to have.

    Reads only the bounding window, so this is cheap enough to call per
    location; the raster is never loaded whole. Windows crossing the
    antimeridian are read as two pieces and summed.
    """
    from rasterio.windows import Window

    close = src is None
    src = src or _open()
    try:
        dlat = radius_km / 110.574
        coslat = max(math.cos(math.radians(lat)), 1e-6)
        dlon = radius_km / (111.320 * coslat)
        if dlon >= 180.0:                      # polar cap: whole lon range
            spans = [(-180.0, 180.0)]
        else:
            w, e = lon - dlon, lon + dlon
            spans = ([(w, e)] if -180.0 <= w and e <= 180.0
                     else [(-180.0, ((e + 180.0) % 360.0) - 180.0),
                           (((w + 180.0) % 360.0) - 180.0, 180.0)])

        south = max(lat - dlat, -90.0)
        north = min(lat + dlat, 90.0)
        tf = src.transform
        x0, y0 = tf.c, tf.f                    # upper-left corner
        px, py = tf.a, -tf.e                   # pixel size (py > 0)
        nodata = src.nodata

        total = 0.0
        for west, east in spans:
            c0 = max(int(math.floor((west - x0) / px)), 0)
            c1 = min(int(math.ceil((east - x0) / px)), _globe_cols(src))
            r0 = max(int(math.floor((y0 - north) / py)), 0)
            r1 = min(int(math.ceil((y0 - south) / py)), src.height)
            if c1 <= c0 or r1 <= r0:
                continue
            a = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0),
                         masked=False).astype("float64")
            if nodata is not None:
                a[a == nodata] = 0.0
            a[~np.isfinite(a)] = 0.0
            a[a < 0.0] = 0.0                   # GHSL uses negative fill
            plat = y0 - (np.arange(r0, r1) + 0.5) * py
            plon = x0 + (np.arange(c0, c1) + 0.5) * px
            d = _haversine_km(lat, lon, plat[:, None], plon[None, :])
            total += float(a[d <= radius_km].sum())
        return int(round(total))
    finally:
        if close:
            src.close()


# ------------------------------------------------------------ grid aggregation

def _lat_index(plat: np.ndarray) -> np.ndarray:
    """POWER latitude row for each pixel-centre latitude.

    Cell i is centred at -90 + 0.5i and spans +/-0.25 around that, so the
    two polar rows are half-height. Counts are summed, so a half-height cell
    is not a bias -- it is simply a smaller cell.
    """
    return np.clip(np.floor((plat + 90.0 + DLAT / 2.0) / DLAT).astype("int64"),
                   0, N_LAT - 1)


def _lon_index(plon: np.ndarray) -> np.ndarray:
    """POWER longitude column for each pixel-centre longitude (wraps)."""
    return (np.floor((plon + 180.0 + DLON / 2.0) / DLON).astype("int64")
            % N_LON)


def build_power_grid(chunk_rows: int = 512,
                     verbose: bool = True) -> np.ndarray:
    """Sum GHS-POP onto the POWER grid; write the .npy and its JSON sidecar.

    Streams the raster in row chunks (~90 MB each) and accumulates, so peak
    memory stays flat regardless of source size. The column fold is checked
    against a brute-force per-pixel bincount on the first chunk before it is
    used for the remaining 21,000 rows.
    """
    lat, lon = power_axes()
    src = _open()
    try:
        tf = src.transform
        x0, y0, px, py = tf.c, tf.f, tf.a, -tf.e
        nodata = src.nodata
        if verbose:
            print(f"raster  {src.width} x {src.height}  origin ({x0}, {y0})  "
                  f"res {px:.8f}  nodata {nodata}  dtype {src.dtypes[0]}")

        use_w = _globe_cols(src)
        if verbose and use_w != src.width:
            print(f"trimming {src.width - use_w} wrap column(s): reading "
                  f"{use_w} cols = 360 deg covered exactly once")
        col_idx = _lon_index(x0 + (np.arange(use_w) + 0.5) * px)
        # fast path: contiguous fold, verified below
        per_cell = use_w // N_LON
        exact = (use_w % N_LON == 0
                 and abs(px * per_cell - DLON) < 1e-6)
        shift = 0
        if exact:
            zeros = np.flatnonzero(col_idx == 0)
            # cell 0 wraps the seam: its pixels are a tail block + a head block
            gaps = np.flatnonzero(np.diff(zeros) > 1)
            if len(gaps) == 1:
                shift = use_w - int(zeros[gaps[0] + 1])
            elif len(gaps) == 0:
                shift = -int(zeros[0])
            else:
                exact = False
        if exact:
            probe = np.roll(col_idx, shift).reshape(N_LON, per_cell)
            exact = bool((probe == np.arange(N_LON)[:, None]).all())
        if verbose:
            print(f"column fold: {'exact' if exact else 'general bincount'}"
                  f"  ({per_cell} px per POWER lon cell, roll {shift})")

        grid = np.zeros((N_LAT, N_LON), dtype="float64")
        checked = False
        for r0 in range(0, src.height, chunk_rows):
            from rasterio.windows import Window
            n = min(chunk_rows, src.height - r0)
            a = src.read(1, window=Window(0, r0, use_w, n),
                         masked=False).astype("float64")
            if nodata is not None:
                a[a == nodata] = 0.0
            a[~np.isfinite(a)] = 0.0
            a[a < 0.0] = 0.0

            if exact:
                cols = np.roll(a, shift, axis=1).reshape(
                    n, N_LON, per_cell).sum(axis=2)
            else:
                cols = np.stack([np.bincount(col_idx, weights=row,
                                             minlength=N_LON) for row in a])
            if not checked:                     # prove the fold, once
                ref = np.stack([np.bincount(col_idx, weights=row,
                                            minlength=N_LON) for row in a[:4]])
                err = float(np.abs(ref - cols[:4]).max())
                if err > 1e-6:
                    raise SystemExit(
                        f"column fold disagrees with bincount by {err} -- "
                        "geometry assumption is wrong, refusing to build")
                if verbose:
                    print(f"fold verified against bincount (max diff {err:g})")
                checked = True

            plat = y0 - (np.arange(r0, r0 + n) + 0.5) * py
            np.add.at(grid, _lat_index(plat), cols)
            if verbose and (r0 // chunk_rows) % 8 == 0:
                print(f"  rows {r0:6d}/{src.height}  "
                      f"running total {grid.sum() / 1e9:.3f}B", flush=True)
    finally:
        src.close()

    out = grid.astype("float32")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(GRID_NPY, out)
    total = float(grid.sum())

    def at(la: float, lo: float) -> float:
        return float(out[int(np.argmin(np.abs(lat - la))),
                         int(np.argmin(np.abs(lon - lo)))])

    spots = {"dhaka_23.75_90.4": at(23.75, 90.4),
             "sahara_23.0_13.0": at(23.0, 13.0),
             "mid_pacific_0_-140": at(0.0, -140.0)}
    meta = {
        "source": SOURCE,
        "grid": {"shape": [N_LAT, N_LON], "lat_deg": DLAT, "lon_deg": DLON,
                 "lat_range": [float(lat[0]), float(lat[-1])],
                 "lon_range": [float(lon[0]), float(lon[-1])],
                 "axes_from": "global_ladders.npz" if LADDERS.exists()
                 else "constructed (MERRA-2 geometry)"},
        "units": "people per POWER cell",
        "aggregation": "sum of GHS-POP counts, pixel centre in cell",
        "global_total": total,
        "spot_checks": spots,
        "built_by": "pipelines/population.py",
    }
    GRID_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nglobal total {total:,.0f}  ({total / 1e9:.3f} billion)")
        band = "PASS" if 7.5e9 <= total <= 8.5e9 else "OUT OF BAND"
        print(f"sanity band 7.5-8.5B -> {band}")
        for k, v in spots.items():
            print(f"  {k:<22} {v:15,.0f}")
        print(f"wrote {GRID_NPY} and {GRID_JSON}")
    return out


# ------------------------------------------------------------------- the demo

def demo(top: int = 5) -> None:
    """Load the grid and print the most-populated POWER cells."""
    if not GRID_NPY.exists():
        raise SystemExit(f"missing {GRID_NPY} -- run --build first")
    g = np.load(GRID_NPY)
    lat, lon = power_axes()
    meta = json.loads(GRID_JSON.read_text(encoding="utf-8")) \
        if GRID_JSON.exists() else {}
    total = float(g.sum())
    print(f"{meta.get('source', {}).get('product', 'population grid')}")
    print(f"grid {g.shape}  total {total:,.0f} ({total / 1e9:.3f}B)\n")
    flat = np.argsort(g, axis=None)[::-1][:top]
    print(f"{'#':<3}{'lat':>8}{'lon':>10}{'people':>16}")
    for n, f in enumerate(flat, 1):
        i, j = np.unravel_index(f, g.shape)
        print(f"{n:<3}{lat[i]:>8.2f}{lon[j]:>10.3f}{g[i, j]:>16,.0f}")


def download() -> None:
    """Print the one-time fetch; the download is run by hand, sequentially."""
    print("GHS-POP R2023A 30 arcsec global, EPSG:4326 (~482 MB):\n"
          f"  {SOURCE['url']}\n"
          f"unzip the .tif into {POP_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true",
                    help="build the POWER grid")
    ap.add_argument("--demo", action="store_true",
                    help="top-5 populated cells")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--near", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--radius", type=float, default=10.0)
    a = ap.parse_args()
    if a.download:
        download()
    if a.build:
        build_power_grid()
    if a.near:
        n = people_near(a.near[0], a.near[1], a.radius)
        print(f"{n:,} people within {a.radius:g} km of "
              f"({a.near[0]}, {a.near[1]})")
    if a.demo:
        demo()
    if not any((a.build, a.demo, a.download, a.near)):
        ap.print_help()
