"""Assemble the two training matrices for the landslide hazard.

  susceptibility.csv -- terrain features, background-sampled negatives, spatial
                        blocks. Answers "where can this happen at all".
  trigger.csv        -- rainfall-anomaly features, case-crossover controls at
                        the same location on season-matched dates. Answers
                        "given a susceptible place, is today unusual".

Run:  python features/build_dataset.py --stage all
Stages are separable so a slow network step can be resumed without redoing the
rest. Everything is cached on disk, so re-runs are cheap.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, PROCESSED  # noqa: E402
from features import sampling  # noqa: E402
from features import terrain  # noqa: E402
from pipelines import dem, nasapower, openmeteo  # noqa: E402

WEATHER = {"nasapower": nasapower, "openmeteo": openmeteo}

# Location accuracy we trust for each layer. Terrain varies over tens of metres,
# so a "25km" point has a meaningless slope; ERA5-Land is ~11 km, so 5 km is
# already inside one pixel and costs nothing there.
ACC_TERRAIN = {"exact", "1km"}
ACC_WEATHER = {"exact", "1km", "5km"}

BG_RATIO = 5          # background points per positive
CONTROLS = 4          # control dates per event
MIN_EVENTS_PER_CELL = 2   # weather-budget filter, see build_trigger()
SEED = 17


def region():
    return json.loads((PROCESSED / "region.json").read_text(encoding="utf-8"))


def load_reports(bbox, accuracies, need_date=True):
    x0, y0, x1, y1 = bbox
    out = []
    with (RAW / "coolr_reports_points.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                la, lo = float(r["latitude"]), float(r["longitude"])
            except (TypeError, ValueError):
                continue
            if not (x0 <= lo <= x1 and y0 <= la <= y1):
                continue
            if (r.get("location_accuracy") or "").strip() not in accuracies:
                continue
            d = r.get("event_date_iso")
            if need_date and not d:
                continue
            out.append({"lat": la, "lon": lo, "date": d,
                        "category": r.get("landslide_category"),
                        "size": r.get("landslide_size"),
                        "trigger": r.get("landslide_trigger")})
    return out


def prefetch_dem(points, workers=4):
    names = sorted(dem.tiles_for_points((p["lat"], p["lon"]) for p in points))
    missing = [n for n in names if not dem.tile_path(n).exists()
               and not (dem.DEM_DIR / f"{n}.missing").exists()]
    print(f"  DEM tiles needed: {len(names)}  already cached: {len(names)-len(missing)}")
    if not missing:
        return
    print(f"  downloading {len(missing)} tiles with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(dem.ensure_tile, n): n for n in missing}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:                       # noqa: BLE001
                print(f"    tile {futs[f]} failed: {e}")
            print(f"    tiles {done}/{len(missing)}", end="\r", flush=True)
    print()


def is_land(t: dict) -> bool:
    """Copernicus DSM writes 0 over open water; flat-and-at-zero means ocean."""
    return not (t["elev_m"] <= 0.5 and t["roughness_std"] < 0.1)


# ------------------------------------------------------------ stage: susc ---

def build_susceptibility(bg_mode: str = "target_group"):
    reg = region()
    bbox = reg["bbox"]
    pos_raw = load_reports(bbox, ACC_TERRAIN, need_date=False)
    pos = sampling.dedupe_locations(pos_raw)
    print(f"[susc] positives: {len(pos_raw)} reports -> {len(pos)} distinct ~90m locations")

    n_bg = len(pos) * BG_RATIO
    # See D10/sampling.py: uniform background over this region fills the
    # negatives with remote steep wilderness nobody reports on, and inverts the
    # model. Target-group sampling matches the observation process instead.
    draw = (sampling.target_group_background if bg_mode == "target_group"
            else sampling.background_points)
    bg = draw(bbox, pos, int(n_bg * 1.6), seed=SEED)
    print(f"[susc] background ({bg_mode}) drawn: {len(bg)} (target {n_bg} after land filter)")

    prefetch_dem(pos + bg)

    rows, dropped_sea, dropped_nodem = [], 0, 0
    for label, pts in ((1, pos), (0, bg)):
        kept = 0
        for p in pts:
            if label == 0 and kept >= n_bg:
                break
            t = terrain.derive(p["lat"], p["lon"])
            if t is None:
                dropped_nodem += 1
                continue
            if not is_land(t):
                dropped_sea += 1
                continue
            rows.append({"lat": p["lat"], "lon": p["lon"], "label": label, **t})
            kept += 1
        print(f"[susc]   label={label}: kept {kept}")
    terrain.close_all()
    print(f"[susc] dropped: {dropped_sea} sea, {dropped_nodem} no-DEM")

    sampling.spatial_blocks(rows)
    cols = ["lat", "lon", "label", "block_id"] + terrain.FEATURES
    out = PROCESSED / "susceptibility.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    npos = sum(r["label"] for r in rows)
    print(f"[susc] wrote {len(rows):,} rows ({npos:,} pos) -> {out.name}")
    return {"n_rows": len(rows), "n_pos": npos, "n_bg": len(rows) - npos,
            "features": terrain.FEATURES,
            "base_rate": npos / len(rows) if rows else 0.0}


# --------------------------------------------------------- stage: trigger ---

def build_trigger(source: str = "nasapower", min_events: int = 1):
    wx = WEATHER[source]
    reg = region()
    bbox = reg["bbox"]
    ev = load_reports(bbox, ACC_WEATHER, need_date=True)
    ev = [e for e in ev if e["date"] >= wx.START]
    print(f"[trig] dated events in box within weather span: {len(ev):,}")

    # group events by weather cell so controls know the local event history
    by_cell = defaultdict(list)
    for e in ev:
        by_cell[wx.cell(e["lat"], e["lon"])].append(e)
    print(f"[trig] source={source}  distinct weather cells: {len(by_cell):,}")

    # Concentrate the (rate-limited) weather budget on cells that actually
    # carry repeat events. Dropping single-event cells removes 47% of the
    # fetch volume while keeping ~85% of the labels -- and a cell with one
    # event contributes one stratum, so the loss per cell fetched is small.
    keep = {c for c, g in by_cell.items() if len(g) >= min_events}
    ev = [e for e in ev if wx.cell(e["lat"], e["lon"]) in keep]
    by_cell = {c: g for c, g in by_cell.items() if c in keep}
    print(f"[trig] cells with >={min_events} events: {len(keep):,} "
          f"-> {len(ev):,} events retained")

    cases = []
    for e in ev:
        c = wx.cell(e["lat"], e["lon"])
        hist = [x["date"] for x in by_cell[c]]
        cases.append({"lat": e["lat"], "lon": e["lon"], "date": e["date"],
                      "label": 1, "stratum": f'{e["lat"]:.4f}_{e["lon"]:.4f}_{e["date"]}'})
        # Controls must fall inside the weather record, and clear of the
        # leading window needed for a 30-day antecedent sum.
        for cd in sampling.control_dates(e["date"], hist, n=CONTROLS, seed=SEED,
                                         year_min=int(wx.START[:4]) + 1,
                                         year_max=int(wx.END[:4])):
            cases.append({"lat": e["lat"], "lon": e["lon"], "date": cd, "label": 0,
                          "stratum": f'{e["lat"]:.4f}_{e["lon"]:.4f}_{e["date"]}'})
    print(f"[trig] case-crossover rows: {len(cases):,} "
          f"({sum(c['label'] for c in cases):,} cases)")

    cells = sorted({wx.cell(c["lat"], c["lon"]) for c in cases})
    print(f"[trig] weather cells: {len(cells)}", flush=True)
    wx.fetch_cells(cells)

    rows, miss = [], 0
    for c in cases:
        f = wx.features_at(c["lat"], c["lon"], c["date"])
        if f is None:
            miss += 1
            continue
        rows.append({k: c[k] for k in ("lat", "lon", "date", "label", "stratum")} | f)
    print(f"[trig] rows with weather: {len(rows):,}  (dropped {miss:,})")

    # a stratum is only usable if it kept its case AND at least one control
    ok = {s for s, g in
          ((s, [r for r in rows if r["stratum"] == s]) for s in {r["stratum"] for r in rows})
          if any(r["label"] == 1 for r in g) and any(r["label"] == 0 for r in g)}
    rows = [r for r in rows if r["stratum"] in ok]
    print(f"[trig] complete strata: {len(ok):,} -> {len(rows):,} rows")

    sampling.spatial_blocks(rows)
    # The weather cell is the real unit of independence for this layer: every
    # row inside one cell shares the same rainfall series, so splitting a cell
    # across folds would put near-identical features on both sides. It is
    # coarser than the 0.25deg block, so it is the correct (stricter) group.
    for r in rows:
        c = wx.cell(r["lat"], r["lon"])
        r["wx_cell"] = f"{c[0]}_{c[1]}"
    cols = ["lat", "lon", "date", "label", "stratum", "block_id", "wx_cell"] + list(wx.FEATURES)
    out = PROCESSED / "trigger.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    npos = sum(r["label"] for r in rows)
    print(f"[trig] wrote {len(rows):,} rows ({npos:,} cases) -> {out.name}")
    return {"n_rows": len(rows), "n_pos": npos, "n_bg": len(rows) - npos,
            "features": list(wx.FEATURES), "weather_source": source,
            "base_rate": npos / len(rows) if rows else 0.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["susceptibility", "trigger", "all"], default="all")
    ap.add_argument("--bg", choices=["target_group", "uniform"], default="target_group",
                    help="background sampling design for the susceptibility layer")
    ap.add_argument("--weather", choices=list(WEATHER), default="nasapower",
                    help="rainfall source for the trigger layer")
    ap.add_argument("--min-events", type=int, default=1,
                    help="drop weather cells with fewer events than this")
    a = ap.parse_args()

    man = {}
    if a.stage in ("susceptibility", "all"):
        man["susceptibility"] = build_susceptibility(a.bg)
    if a.stage in ("trigger", "all"):
        man["trigger"] = build_trigger(a.weather, a.min_events)

    p = PROCESSED / "features_manifest.json"
    prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if "layers" not in prev:
        prev = {"layers": {}}
    prev["layers"].update(man)
    tot = prev["layers"]
    prev.update({
        "n_rows": sum(v["n_rows"] for v in tot.values()),
        "n_pos": sum(v["n_pos"] for v in tot.values()),
        "n_bg": sum(v["n_bg"] for v in tot.values()),
        "base_rate": (sum(v["n_pos"] for v in tot.values())
                      / max(1, sum(v["n_rows"] for v in tot.values()))),
        "features": sorted({f for v in tot.values() for f in v["features"]}),
    })
    p.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {p}")
