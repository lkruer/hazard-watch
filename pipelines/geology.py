"""Geology and soil substrate features for the susceptibility layer.

Terrain (features/terrain.py) answers "what shape is this ground". This module
answers "what is the ground MADE OF" -- the missing half of landslide physics.
D24 named the gap directly: Oso 2014 was a deep-seated failure in glacial
outwash, invisible to surface slope, and lithology was put on the roadmap.

Two sources, both public/keyless/open-licensed (the project's hard constraint):

  SoilGrids 250m v2.0  ISRIC, CC-BY 4.0.
      Texture (clay/sand/silt), bulk density, coarse fragments. Fetched as
      GeoTIFF windows over WCS 2.0.1 from maps.isric.org -- NOT the point REST
      API, which is rate-limited and would need tens of thousands of calls.
      Citation: Poggio et al. (2021), SOIL 7, 217-240.

  GLiM v1.0            Hartmann & Moosdorf (2012), CC-BY 3.0, PANGAEA
      doi:10.1594/PANGAEA.788537. Global lithology, 16 classes.
      *** RESOLUTION WARNING, read before trusting lith_class ***
      The openly-archived GLiM is the 0.5-degree GRIDDED version (720x360),
      i.e. ~55 km cells. The full 1,235,400-polygon shapefile is distributed
      via CCGM.ORG, a commercial publisher, so it is not reachable under the
      keyless/open constraint. At 0.5 deg the Myanmar box holds ~16 cells and
      is nearly constant; this feature is a coarse regional stratifier, not a
      site-scale geology map. Treated as a documented compromise, not a
      silent one.

WCS mechanics that cost time, recorded so the next person skips them:
  * Native grid is Interrupted Goode Homolosine (EPSG:152160) at 250 m, and a
    native-CRS GetCoverage returns a GeoTIFF with NO CRS tag. Passing
    SUBSETTINGCRS+OUTPUTCRS=EPSG:4326 makes the server reproject and hand back
    a properly-tagged lat/lon raster whose bounds match the request exactly.
  * SUBSET axis labels are X (longitude) and Y (latitude) in that CRS.
  * There is no 0-30cm coverage. The three standard slices 0-5/5-15/15-30 are
    combined here by depth weighting (5/30, 10/30, 15/30).
  * clay+sand+silt sums to 1000 g/kg (checked: mean 1000.0, range 999-1001),
    so silt is DERIVED rather than downloaded -- three fewer coverages.

Units: SoilGrids ships integers. clay/sand/silt g/kg -> % (/10); bdod cg/cm3
-> g/cm3 (/100); cfvo cm3/dm3 -> vol% (/10).

Tiling: a fixed 2-degree grid, fetching only tiles that actually contain
points. The regions' bounding boxes are misleading -- Brazil's spans 19x22 deg
but its points sit in 13 one-degree cells -- so a per-bbox window would have
downloaded mostly ocean. 18 tiles cover all three pilot regions.
"""
from __future__ import annotations

import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

GEO_DIR = CACHE / "geology"
SOIL_DIR = GEO_DIR / "soilgrids"
GLIM_DIR = GEO_DIR / "glim"
for _d in (GEO_DIR, SOIL_DIR, GLIM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

WCS_BASE = "https://maps.isric.org/mapserv?map=/map/{prop}.map"
TILE_DEG = 2.0

# depth slices making up 0-30 cm, with their thickness weights
DEPTHS = [("0-5cm", 5.0), ("5-15cm", 10.0), ("15-30cm", 15.0)]
DEPTH_TOTAL = sum(w for _, w in DEPTHS)

# property -> (divisor to physical units, output column)
PROPS = {
    "clay": (10.0, "clay_pct"),        # g/kg   -> %
    "sand": (10.0, "sand_pct"),        # g/kg   -> %
    "bdod": (100.0, "bdod_g_cm3"),     # cg/cm3 -> g/cm3
    "cfvo": (10.0, "cfvo_pct"),        # cm3/dm3-> vol%
}

FEATURES = ["clay_pct", "silt_pct", "sand_pct", "bdod_g_cm3", "cfvo_pct",
            "lith_class"]
SOIL_FEATURES = [f for f in FEATURES if f != "lith_class"]

# GLiM class codes. Integer values are the Value_ column of the archive's
# Classnames.txt, kept verbatim so the mapping is auditable against the source.
GLIM_CLASSES = {
    1: "su",   # unconsolidated sediments
    2: "vb",   # basic volcanic rocks
    3: "ss",   # siliciclastic sedimentary rocks
    4: "pb",   # basic plutonic rocks
    5: "sm",   # mixed sedimentary rocks
    6: "sc",   # carbonate sedimentary rocks
    7: "va",   # acid volcanic rocks
    8: "mt",   # metamorphic rocks
    9: "pa",   # acid plutonic rocks
    10: "vi",  # intermediate volcanic rocks
    11: "wb",  # water bodies
    12: "py",  # pyroclastics
    13: "pi",  # intermediate plutonic rocks
    14: "ev",  # evaporites
    15: "nd",  # no data
    16: "ig",  # ice and glaciers
}
GLIM_LONG = {
    "su": "unconsolidated sediments", "vb": "basic volcanic rocks",
    "ss": "siliciclastic sedimentary rocks", "pb": "basic plutonic rocks",
    "sm": "mixed sedimentary rocks", "sc": "carbonate sedimentary rocks",
    "va": "acid volcanic rocks", "mt": "metamorphic rocks",
    "pa": "acid plutonic rocks", "vi": "intermediate volcanic rocks",
    "wb": "water bodies", "py": "pyroclastics",
    "pi": "intermediate plutonic rocks", "ev": "evaporites",
    "nd": "no data", "ig": "ice and glaciers",
}
GLIM_URL = "https://hdl.handle.net/10013/epic.39939.d001"
GLIM_ZIP = GLIM_DIR / "hartmann-moosdorf_2012.zip"
GLIM_ASC = "glim_wgs84_0point5deg.txt.asc"


# ------------------------------------------------------------------ tiles ---

def tile_sw(lat: float, lon: float) -> tuple[int, int]:
    """South-west corner of the TILE_DEG tile containing the point."""
    return (int(math.floor(lat / TILE_DEG) * TILE_DEG),
            int(math.floor(lon / TILE_DEG) * TILE_DEG))


def tile_key(la: int, lo: int) -> str:
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"{ns}{abs(la):02d}_{ew}{abs(lo):03d}"


def soil_tile_path(prop: str, depth: str, la: int, lo: int) -> Path:
    return SOIL_DIR / f"{prop}_{depth}_{tile_key(la, lo)}.tif"


def ensure_soil_tile(prop: str, depth: str, la: int, lo: int,
                     polite: float = 0.4) -> Path | None:
    """Download one SoilGrids WCS window, cached. None if the server has no
    coverage there (all-ocean tiles can come back empty)."""
    p = soil_tile_path(prop, depth, la, lo)
    if p.exists() and p.stat().st_size > 0:
        return p
    miss = p.with_suffix(".missing")
    if miss.exists():
        return None

    params = {
        "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage",
        "COVERAGEID": f"{prop}_{depth}_mean", "FORMAT": "image/tiff",
        "SUBSET": [f"X({lo},{lo + TILE_DEG:g})", f"Y({la},{la + TILE_DEG:g})"],
        "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
    }
    r = SESSION.get(WCS_BASE.format(prop=prop), params=params, timeout=600)
    if r.status_code != 200 or r.content[:2] not in (b"II", b"MM"):
        head = r.text[:200].replace("\n", " ") if r.content[:2] not in (b"II", b"MM") else ""
        print(f"    {prop}_{depth} {tile_key(la, lo)}: no coverage "
              f"({r.status_code}) {head}")
        miss.write_text(f"{r.status_code}", encoding="utf-8")
        return None
    tmp = p.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(p)
    time.sleep(polite)          # public service, one request at a time
    return p


# ------------------------------------------------------------- tile bundle --

_BUNDLE: dict[tuple[int, int], dict | None] = {}


def _read_depth_stack(prop: str, la: int, lo: int):
    """Depth-weighted 0-30 cm mean for one property, in raw integer units."""
    acc = None
    wsum = 0.0
    transform = shape = None
    for depth, w in DEPTHS:
        p = ensure_soil_tile(prop, depth, la, lo)
        if p is None:
            continue
        with rasterio.open(p) as ds:
            a = ds.read(1).astype("float64")
            if transform is None:
                transform, shape = ds.transform, a.shape
            elif a.shape != shape:
                # depth slices are requested with identical bounds, so this
                # should not happen; guard rather than silently misalign
                continue
        acc = a * w if acc is None else acc + a * w
        wsum += w
    if acc is None or wsum == 0:
        return None, None
    return acc / wsum, transform


def tile_bundle(la: int, lo: int) -> dict | None:
    """All soil properties for one tile, physical units, NaN where unmapped.

    Validity comes from texture: SoilGrids writes 0 where nothing is mapped
    (water, ice, bare rock). clay and sand cannot both be 0 in real soil, so
    that pair defines the mask. cfvo genuinely CAN be 0 (no coarse fragments),
    which is why it is masked by texture rather than by its own zeros.
    """
    key = (la, lo)
    if key in _BUNDLE:
        return _BUNDLE[key]

    raw = {}
    transform = None
    for prop in PROPS:
        a, tr = _read_depth_stack(prop, la, lo)
        if a is None:
            continue
        raw[prop] = a
        transform = transform if transform is not None else tr
    if "clay" not in raw or "sand" not in raw:
        _BUNDLE[key] = None
        return None

    valid = (raw["clay"] > 0) & (raw["sand"] > 0)
    out = {"transform": transform, "shape": raw["clay"].shape, "valid": valid}
    for prop, (div, col) in PROPS.items():
        if prop not in raw:
            out[col] = np.full(raw["clay"].shape, np.nan, dtype="float32")
            continue
        v = raw[prop] / div
        out[col] = np.where(valid, v, np.nan).astype("float32")
    # silt is the compositional remainder, not a separate download
    out["silt_pct"] = np.where(
        valid, 100.0 - out["clay_pct"] - out["sand_pct"], np.nan).astype("float32")
    _BUNDLE[key] = out
    return out


def clear_cache() -> None:
    """Drop in-memory tile arrays. One 2-deg tile is ~16 MB across the five
    float32 grids, so annotating region by region beats holding all 18."""
    _BUNDLE.clear()


# -------------------------------------------------------------------- GLiM --

_GLIM: dict | None = None


def ensure_glim() -> Path | None:
    if GLIM_ZIP.exists() and GLIM_ZIP.stat().st_size > 0:
        return GLIM_ZIP
    r = SESSION.get(GLIM_URL, timeout=600)
    if r.status_code != 200:
        print(f"    GLiM download failed ({r.status_code})")
        return None
    GLIM_ZIP.write_bytes(r.content)
    return GLIM_ZIP


def glim_grid() -> dict | None:
    """The 0.5-degree lithology grid as an in-memory array."""
    global _GLIM
    if _GLIM is not None:
        return _GLIM
    p = ensure_glim()
    if p is None:
        return None
    with zipfile.ZipFile(p) as z:
        txt = z.read(GLIM_ASC).decode("utf-8", "replace")
    lines = txt.splitlines()
    hdr = {}
    i = 0
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2 and not parts[0].lstrip("-").isdigit():
            hdr[parts[0].lower()] = float(parts[1])
        else:
            break
    a = np.loadtxt(lines[i:], dtype="float64")
    nod = hdr.get("nodata_value", -9999.0)
    a[a == nod] = np.nan
    _GLIM = {"a": a, "ncols": int(hdr["ncols"]), "nrows": int(hdr["nrows"]),
             "xll": hdr["xllcorner"], "yll": hdr["yllcorner"],
             "cell": hdr["cellsize"]}
    return _GLIM


def lith_class(lat: float, lon: float) -> float:
    """GLiM class integer (see GLIM_CLASSES), NaN off-grid / no data."""
    g = glim_grid()
    if g is None:
        return float("nan")
    col = int((lon - g["xll"]) // g["cell"])
    # ESRI ASCII rows run north -> south
    row = int((g["yll"] + g["nrows"] * g["cell"] - lat) // g["cell"])
    if not (0 <= col < g["ncols"] and 0 <= row < g["nrows"]):
        return float("nan")
    v = g["a"][row, col]
    return float(v) if np.isfinite(v) else float("nan")


# ---------------------------------------------------------------- sampling --

def geo_features(lat: float, lon: float) -> dict:
    """Soil + lithology at a point. Values are NaN where a source has no data.

    Returns keys: clay_pct, silt_pct, sand_pct, bdod_g_cm3, cfvo_pct,
    lith_class.
    """
    out = {f: float("nan") for f in FEATURES}
    out["lith_class"] = lith_class(lat, lon)

    la, lo = tile_sw(lat, lon)
    b = tile_bundle(la, lo)
    if b is None:
        return out
    tr = b["transform"]
    col, row = ~tr * (lon, lat)
    row, col = int(row), int(col)
    h, w = b["shape"]
    if not (0 <= row < h and 0 <= col < w):
        row = min(max(row, 0), h - 1)
        col = min(max(col, 0), w - 1)
    for f in SOIL_FEATURES:
        v = b[f][row, col]
        out[f] = float(v) if np.isfinite(v) else float("nan")
    return out


def prefetch(points, label: str = "") -> None:
    """Download every tile needed by an iterable of (lat, lon), sequentially."""
    tiles = sorted({tile_sw(la, lo) for la, lo in points})
    n = len(tiles) * len(PROPS) * len(DEPTHS)
    print(f"  [{label}] {len(tiles)} tiles x {len(PROPS)} props x "
          f"{len(DEPTHS)} depths = {n} windows")
    done = 0
    for la, lo in tiles:
        for prop in PROPS:
            for depth, _ in DEPTHS:
                ensure_soil_tile(prop, depth, la, lo)
                done += 1
        got = sum(1 for prop in PROPS for depth, _ in DEPTHS
                  if soil_tile_path(prop, depth, la, lo).exists())
        print(f"    {tile_key(la, lo)}  {got}/{len(PROPS)*len(DEPTHS)} windows"
              f"   [{done}/{n}]")


def cache_bytes() -> int:
    return sum(p.stat().st_size for p in GEO_DIR.rglob("*") if p.is_file())


if __name__ == "__main__":
    probes = [
        (48.2836, -121.8477, "Oso WA (deep-seated, glacial outwash)"),
        (45.52, -122.68, "Portland OR (valley floor)"),
        (22.5, 93.0, "Chin Hills, Myanmar"),
        (-29.2, -51.5, "Serra Gaucha, Brazil"),
    ]
    for la, lo, what in probes:
        d = geo_features(la, lo)
        code = GLIM_CLASSES.get(int(d["lith_class"])) if np.isfinite(d["lith_class"]) else None
        print(f"\n{what}  ({la}, {lo})")
        for f in SOIL_FEATURES:
            print(f"  {f:<14} {d[f]:8.2f}")
        print(f"  {'lith_class':<14} {d['lith_class']:8.0f}"
              f"  {code} = {GLIM_LONG.get(code, '?')}")
    print(f"\ncache: {cache_bytes()/1e6:.1f} MB under {GEO_DIR}")
