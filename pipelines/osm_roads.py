"""Distance-to-nearest-arterial-road from OpenStreetMap, as a bias control.

Why this feature exists (docs/decisions.md D10/D11): COOLR labels are made by
people, and people are near roads. Target-group background sampling cancelled
most of that in the class balance, but elevation still proxies what remains --
valley floors are where the roads are. Making accessibility EXPLICIT lets the
model attribute the reporting artifact to `road_dist_m` instead of smuggling
it through elevation, after which the terrain features carry terrain.

The road network is the arterial skeleton (motorway..tertiary) from Overpass,
fetched in lat chunks by the session script into data/cache/osm/roads_*.json.gz.
Residential/track roads are deliberately excluded: volume is 10x and the
accessibility *gradient* -- how far civilisation is -- is carried fine by the
arterials.

Distances are computed on ECEF unit vectors with a cKDTree (exact chord
distance converted to arc), so there is no flat-earth error over a 7-degree
box. Vertex spacing on OSM arterials (tens of metres) is finer than the 30 m
DEM cell, so nearest-vertex ~= nearest-point-on-road at our scale.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402

OSM = CACHE / "osm"
EARTH_R = 6_371_000.0

_TREE: cKDTree | None = None
_N_VERTS = 0


def _ecef(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def load_tree() -> cKDTree:
    global _TREE, _N_VERTS
    if _TREE is not None:
        return _TREE
    chunks = sorted(OSM.glob("roads_*.json.gz"))
    if not chunks:
        raise FileNotFoundError(f"no OSM road chunks in {OSM}")
    pts = []
    for p in chunks:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
        for el in d.get("elements", []):
            for g in el.get("geometry", ()) or ():
                pts.append((g["lat"], g["lon"]))
    a = np.asarray(pts, dtype="float64")
    _N_VERTS = len(a)
    _TREE = cKDTree(_ecef(a[:, 0], a[:, 1]))
    print(f"road index: {len(chunks)} chunks, {_N_VERTS:,} vertices")
    return _TREE


def road_dist_m(lats, lons) -> np.ndarray:
    """Great-circle metres from each point to the nearest arterial vertex."""
    tree = load_tree()
    q = _ecef(np.asarray(lats, dtype="float64"),
              np.asarray(lons, dtype="float64"))
    chord, _ = tree.query(q, k=1)
    return 2.0 * EARTH_R * np.arcsin(np.clip(chord / 2.0, 0, 1))


def annotate_susceptibility() -> None:
    """Add road_dist_m to the susceptibility matrix in place."""
    import pandas as pd
    from config import PROCESSED
    p = PROCESSED / "susceptibility.csv"
    df = pd.read_csv(p)
    df["road_dist_m"] = road_dist_m(df["lat"].to_numpy(), df["lon"].to_numpy())
    df.to_csv(p, index=False)
    pos = df[df.label == 1]["road_dist_m"]
    bg = df[df.label == 0]["road_dist_m"]
    print(f"annotated {len(df):,} rows -> {p.name}")
    print(f"  road_dist_m median: positives {pos.median():,.0f} m, "
          f"background {bg.median():,.0f} m")


if __name__ == "__main__":
    annotate_susceptibility()
