"""Render world hazard-percentile fields to PNG maps.

Consumes serve/score_world.py output for a date and draws one map per hazard
signal -- matplotlib only, no basemap dependency; coastlines emerge from the
data itself (ocean cells are NaN in the ladders). A model deliverable for
checking the fields against known events, not a website.

Usage: python reports/render_world.py --date 2024-11-15
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, REPORTS  # noqa: E402

PANELS = [
    ("rain3d_pctl", "3-day rainfall percentile (landslide/flash trigger)",
     "viridis", None),
    ("kbdi_pctl", "KBDI fuel-dryness percentile (fire)", "YlOrRd", None),
    ("vpd_pctl", "Vapor-pressure-deficit percentile (fire)", "YlOrRd", None),
    ("spi90_pctl", "3-month precipitation percentile (drought: LOW = dry)",
     "BrBG", None),
]


def main(date: str) -> None:
    src = PROCESSED / "world" / date
    if not src.exists():
        raise SystemExit(f"run serve/score_world.py --date {date} first")
    z = np.load(PROCESSED / "global_ladders.npz")
    lat, lon = z["lat"], z["lon"]
    extent = [float(lon.min()), float(lon.max()),
              float(lat.min()), float(lat.max())]

    fig, axes = plt.subplots(len(PANELS), 1,
                             figsize=(13, 5.4 * len(PANELS)), dpi=110)
    for ax, (stem, title, cmap, vlim) in zip(np.atleast_1d(axes), PANELS):
        f = src / f"{stem}.npy"
        if not f.exists():
            ax.set_axis_off()
            continue
        a = np.load(f).astype("float32")
        im = ax.imshow(a, origin="lower", extent=extent, cmap=cmap,
                       vmin=0.0, vmax=1.0, aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"{title} — {date}", fontsize=12)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    out = REPORTS / f"world_{date}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    main(ap.parse_args().date)
