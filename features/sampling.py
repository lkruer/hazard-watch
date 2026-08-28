"""Negative sampling for a presence-only hazard catalogue.

COOLR records only where a landslide DID happen. There is no "nothing happened
here" record, so negatives have to be constructed -- and how they are
constructed decides what the model actually learns. This is the single easiest
place to manufacture a great-looking, worthless PR-AUC, so the reasoning is
written down rather than buried.

Two different negative designs, one per layer:

SUSCEPTIBILITY (spatial).  Background points drawn across the study region,
excluding a buffer around known events. Standard species-distribution-modelling
practice (the brief points at MaxEnt for exactly this).
  Known bias, stated rather than hidden: COOLR is media/report-derived, so
  events are over-reported near roads and towns. A model trained this way partly
  learns "where do people notice landslides", not only "where do landslides
  happen". Mitigations: keep the region compact so reporting intensity is more
  uniform, and treat the susceptibility number as relative within-region rather
  than an absolute probability.

TRIGGER (temporal, case-crossover).  For each real event, controls are the SAME
location on OTHER dates. Terrain, geology, land cover, road access and
reporting intensity are all identical within a stratum, so they cannot be used
to separate case from control -- the only thing that varies is weather. This
removes the spatial-bias problem completely for the trigger layer.
  Controls are season-matched (same time of year, different years) so the model
  cannot cheat by learning "winter" -- which would be a real but useless signal,
  since we already know the wet season is wetter.

Sampling is seeded and deterministic: same seed, same dataset.
"""
from __future__ import annotations

import datetime as dt
import math
import random

EARTH_R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def snap(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    return (int(math.floor(lat / cell_deg)), int(math.floor(lon / cell_deg)))


def dedupe_locations(points, cell_deg: float = 0.00083):
    """Collapse points to one per ~90 m cell.

    Repeat reports of the same slope would otherwise be counted as independent
    positives, inflating both the positive count and any metric computed on it.
    """
    seen, out = set(), []
    for p in points:
        k = snap(p["lat"], p["lon"], cell_deg)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def target_group_background(bbox, positives, n, seed=17, min_dist_m=500.0,
                            radius_m=12_000.0):
    """Accessibility-matched background ("target-group") sampling.

    Uniform background over the whole region is WRONG here, and measurably so.
    The Pacific Northwest box contains vast remote steep wilderness that is
    never reported on; uniform sampling fills the negatives with exactly that
    terrain, and the model duly learns "steep and remote = no landslide". The
    first version of this pipeline did that and produced an inverted model --
    the Columbia Gorge (44 deg slope, 481 m relief) scored 0.040 while a flat
    valley floor (17 deg, 53 m) scored 0.196.

    The standard species-distribution-modelling correction is to draw
    background from the SAME observation process that produced the presences,
    so that observer effort cancels instead of being learned. Here that means
    anchoring each background point on a randomly chosen real event and
    offsetting it by up to `radius_m`: the background then shares the
    positives' accessibility footprint, and what remains for the model to learn
    is terrain, not where roads are.

    Points still respect `min_dist_m` from any known event so a "background"
    point is not just an unrecorded part of the same failure.
    """
    x0, y0, x1, y1 = bbox
    rng = random.Random(seed)
    grid_deg = 0.01
    idx: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for p in positives:
        idx.setdefault(snap(p["lat"], p["lon"], grid_deg), []).append((p["lat"], p["lon"]))

    out, tries = [], 0
    while len(out) < n and tries < n * 60:
        tries += 1
        anchor = positives[rng.randrange(len(positives))]
        # uniform in a disc: sqrt for area weighting, so points are not
        # clumped toward the anchor
        r = radius_m * math.sqrt(rng.random())
        th = rng.uniform(0, 2 * math.pi)
        dlat = (r * math.cos(th)) / 111_320.0
        dlon = (r * math.sin(th)) / (111_320.0 * math.cos(math.radians(anchor["lat"])) or 1.0)
        lat, lon = anchor["lat"] + dlat, anchor["lon"] + dlon
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue
        if _too_close(idx, lat, lon, grid_deg, min_dist_m):
            continue
        out.append({"lat": lat, "lon": lon})
    return out


def _too_close(idx, lat, lon, grid_deg, min_dist_m) -> bool:
    gi, gj = snap(lat, lon, grid_deg)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for (pla, plo) in idx.get((gi + di, gj + dj), ()):
                if haversine_m(lat, lon, pla, plo) < min_dist_m:
                    return True
    return False


def background_points(bbox, positives, n, seed=17, min_dist_m=500.0,
                      max_tries_factor=40):
    """Uniform-on-land background points, kept away from known events.

    'On land' is not decided here -- the caller validates each candidate against
    the DEM (ocean has no tile / reads as flat zero) and asks for more if it
    rejects too many. Longitude is sampled uniformly and latitude by
    equal-area (arcsin) so the box is not over-sampled at its northern edge.
    """
    x0, y0, x1, y1 = bbox
    rng = random.Random(seed)
    # index positives on a coarse grid for cheap distance rejection
    grid_deg = 0.01
    idx: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for p in positives:
        idx.setdefault(snap(p["lat"], p["lon"], grid_deg), []).append((p["lat"], p["lon"]))

    s0, s1 = math.sin(math.radians(y0)), math.sin(math.radians(y1))
    out, tries = [], 0
    while len(out) < n and tries < n * max_tries_factor:
        tries += 1
        lat = math.degrees(math.asin(rng.uniform(s0, s1)))
        lon = rng.uniform(x0, x1)
        gi, gj = snap(lat, lon, grid_deg)
        too_close = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for (pla, plo) in idx.get((gi + di, gj + dj), ()):
                    if haversine_m(lat, lon, pla, plo) < min_dist_m:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if not too_close:
            out.append({"lat": lat, "lon": lon})
    return out


def control_dates(event_date: str, all_event_dates_here, n=4, seed=17,
                  season_window=45, exclusion_days=7,
                  year_min=1996, year_max=2024):
    """Season-matched control dates at the same location.

    Same day-of-year +/- `season_window`, drawn from OTHER years, excluding any
    date within `exclusion_days` of a real event at this location (so a control
    cannot secretly be the same storm).
    """
    d0 = dt.date.fromisoformat(event_date)
    rng = random.Random(f"{seed}:{event_date}:{len(all_event_dates_here)}")
    banned = set()
    for e in all_event_dates_here:
        try:
            ed = dt.date.fromisoformat(e)
        except ValueError:
            continue
        for k in range(-exclusion_days, exclusion_days + 1):
            banned.add(ed + dt.timedelta(days=k))

    cands = []
    for yr in range(year_min, year_max + 1):
        if yr == d0.year:
            continue
        try:
            anchor = d0.replace(year=yr)
        except ValueError:                      # 29 Feb
            anchor = d0.replace(year=yr, day=28)
        for off in range(-season_window, season_window + 1):
            c = anchor + dt.timedelta(days=off)
            if c.year < year_min or c.year > year_max:
                continue
            if c in banned:
                continue
            cands.append(c)
    if not cands:
        return []
    rng.shuffle(cands)
    return [c.isoformat() for c in cands[:n]]


def spatial_blocks(rows, block_deg=0.25):
    """Assign a spatial block id, used as the CV group.

    Whole blocks are held out together. A random split would put a point and its
    500 m neighbour on opposite sides of the split -- near-identical features,
    near-identical label -- which leaks and makes the score look far better than
    it is. The brief is explicit about this; GroupKFold on this id, never KFold.
    """
    for r in rows:
        bi, bj = snap(r["lat"], r["lon"], block_deg)
        r["block_id"] = f"{bi}_{bj}"
    return rows
