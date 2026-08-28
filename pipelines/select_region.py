"""Pick the v1 study region empirically instead of by guesswork.

Two label sources with different structure drive two different criteria:
  events  (inventories) -> SUSCEPTIBILITY: want dense, spatially-complete
                           coverage in a compact box. Dates are irrelevant.
  reports (GLC)         -> TRIGGER: want many DISTINCT DATES in the box, since
                           each date is one independent rainfall situation.
Stdlib only.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW  # noqa: E402

CELL = 0.25       # deg, density grid
WIN = 8           # cells per side -> 2.0 deg window


def load(name: str) -> list[dict]:
    rows = list(csv.DictReader((RAW / f"coolr_{name}_points.csv").open(encoding="utf-8")))
    out = []
    for r in rows:
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if lat == 0 and lon == 0:
            continue
        r["_lat"], r["_lon"] = lat, lon
        out.append(r)
    return out


def cell(lat: float, lon: float) -> tuple[int, int]:
    return (int((lat + 90) // CELL), int((lon + 180) // CELL))


def best_windows(rows: list[dict], topn: int = 8, by_dates: bool = False):
    """Slide a WIN x WIN cell window; score by event count or distinct dates."""
    grid: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        grid[cell(r["_lat"], r["_lon"])].append(r)

    scored = []
    seen_anchor = set()
    for (ci, cj) in grid:
        # anchor windows on populated cells only
        for oi in range(0, 1):
            a = (ci, cj)
            if a in seen_anchor:
                continue
            seen_anchor.add(a)
            members, dates = [], set()
            for i in range(ci, ci + WIN):
                for j in range(cj, cj + WIN):
                    for r in grid.get((i, j), ()):
                        members.append(r)
                        if r.get("event_date_iso"):
                            dates.add(r["event_date_iso"])
            if not members:
                continue
            occupied = len({cell(r["_lat"], r["_lon"]) for r in members})
            score = len(dates) if by_dates else len(members)
            scored.append((score, len(members), len(dates), occupied, ci, cj))
    scored.sort(reverse=True)

    # greedy non-overlapping pick
    picked, used = [], []
    for s in scored:
        _, _, _, _, ci, cj = s
        if any(abs(ci - pi) < WIN and abs(cj - pj) < WIN for pi, pj in used):
            continue
        used.append((ci, cj))
        picked.append(s)
        if len(picked) >= topn:
            break
    return picked


def show(title: str, picked, rows_by_cell=None):
    print(f"\n=== {title} ===")
    print(f"  {'events':>7} {'dates':>6} {'cells':>6}   bbox (min_lon,min_lat,max_lon,max_lat)")
    for score, n, nd, occ, ci, cj in picked:
        min_lat = ci * CELL - 90
        min_lon = cj * CELL - 180
        max_lat = min_lat + WIN * CELL
        max_lon = min_lon + WIN * CELL
        print(f"  {n:>7,} {nd:>6,} {occ:>6}   ({min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f})")


if __name__ == "__main__":
    ev = load("events")
    rp = load("reports")
    print(f"usable events: {len(ev):,}   usable reports: {len(rp):,}")

    c = Counter(r.get("country_name") or "(blank)" for r in rp)
    print(f"\nreports countries ({len(c)}): " +
          ", ".join(f"{k}={v}" for k, v in c.most_common(12)))

    show("SUSCEPTIBILITY candidates - densest 2.0deg windows in EVENT inventories",
         best_windows(ev, topn=8))
    show("TRIGGER candidates - most DISTINCT DATES in 2.0deg windows of REPORTS",
         best_windows(rp, topn=8, by_dates=True))
