"""Download every SoilGrids tile the pilot regions need. Sequential, resumable.

Separate from geology.py so the long network job can run unattended while the
experiment code is written; re-running it is a no-op once the cache is warm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED  # noqa: E402
from pipelines import geology as G  # noqa: E402

REGIONS = ["pnw", "myanmar", "brazil"]
OSO = (48.2836, -121.8477)


def main() -> None:
    pts: list[tuple[float, float]] = []
    for r in REGIONS:
        d = pd.read_csv(PROCESSED / f"region_{r}.csv")
        pts += list(zip(d["lat"].tolist(), d["lon"].tolist()))
        print(f"  {r}: {len(d):,} points")
    # the Oso probe and its ~900 m ring must be covered even if no training
    # point lands in that tile
    pts += [OSO]
    for dla in (-0.012, 0.0, 0.012):
        for dlo in (-0.018, 0.0, 0.018):
            pts.append((OSO[0] + dla, OSO[1] + dlo))
    G.prefetch(pts, label="pilot")
    print(f"\ncache: {G.cache_bytes()/1e6:.1f} MB")


if __name__ == "__main__":
    main()
