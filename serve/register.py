"""Register a location: do the heavy, STATIC work exactly once.

The nightly cron must stay light enough for a free CI runner, so everything
slow or large happens here at registration time: DEM tiles + terrain
derivatives, susceptibility (point and ~900 m neighborhood max, Tier A or B),
road distance where the network is cached, flood-basin availability, and
population within 10 km. The nightly job then touches weather only.

IDs are geohashes (precision 7, ~150 m) -- stable, URL-safe, derivable from
the coordinate alone, which is exactly what tap-on-map registration needs.

Usage:
    python serve/register.py add 27.72 85.32 --name "Kathmandu"
    python serve/register.py seed          # demo set spanning the regimes
    python serve/register.py list
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT  # noqa: E402

REGISTRY = ROOT / "serve" / "locations.json"

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat: float, lon: float, precision: int = 7) -> str:
    """Plain-vanilla geohash, no dependencies."""
    lat_r, lon_r = [-90.0, 90.0], [-180.0, 180.0]
    bits, even, ch, out = 0, True, 0, []
    while len(out) < precision:
        if even:
            mid = (lon_r[0] + lon_r[1]) / 2
            if lon >= mid:
                ch = ch * 2 + 1
                lon_r[0] = mid
            else:
                ch = ch * 2
                lon_r[1] = mid
        else:
            mid = (lat_r[0] + lat_r[1]) / 2
            if lat >= mid:
                ch = ch * 2 + 1
                lat_r[0] = mid
            else:
                ch = ch * 2
                lat_r[1] = mid
        even = not even
        bits += 1
        if bits == 5:
            out.append(_B32[ch])
            bits, ch = 0, 0
    return "".join(out)


def _load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"locations": []}


def register(lat: float, lon: float, name: str | None = None) -> dict:
    from features import terrain
    from serve.score_global import _regions, _susc_from_terrain
    from pipelines.glofas import stack_for
    from pipelines.population import people_near

    reg = _load()
    gid = geohash(lat, lon)
    if any(x["id"] == gid for x in reg["locations"]):
        print(f"  {gid} already registered")
        return next(x for x in reg["locations"] if x["id"] == gid)

    t = terrain.derive(lat, lon)
    susc = susc_near = None
    tier = "C"
    region_name = None
    if t is not None:
        hit = next((n for n, r in _regions().items()
                    if r["bbox"][0] <= lon <= r["bbox"][2]
                    and r["bbox"][1] <= lat <= r["bbox"][3]), None)
        region_name, tier = hit, ("A" if hit else "B")
        vals = []
        for dla in (-0.008, 0.0, 0.008):
            for dlo in (-0.008, 0.0, 0.008):
                tn = t if (dla == 0 and dlo == 0) else terrain.derive(
                    lat + dla, lon + dlo)
                if tn is None:
                    continue
                pv = _susc_from_terrain(tn, lat + dla, lon + dlo)
                if pv is not None:
                    vals.append(pv)
                    if dla == 0 and dlo == 0:
                        susc = pv
        susc_near = max(vals) if vals else None

    stack, why = (None, None)
    try:
        stack, why = stack_for(lat, lon)
    except Exception as e:                                  # noqa: BLE001
        why = str(e)
    flood_ok = stack is not None

    entry = {
        "id": gid, "name": name or gid, "lat": lat, "lon": lon,
        "registered": dt.date.today().isoformat(),
        "tier_susceptibility": tier,
        "region_model": region_name,
        "susceptibility": None if susc is None else round(float(susc), 4),
        "susceptibility_nearby_max": (None if susc_near is None
                                      else round(float(susc_near), 4)),
        "terrain": (None if t is None else
                    {"slope_deg": round(t["slope_deg"], 1),
                     "elev_m": round(t["elev_m"], 0),
                     "relief_m": round(t["relief_range"], 0)}),
        "flood_basin_cached": flood_ok,
        "flood_note": None if flood_ok else why,
        "people_10km": int(people_near(lat, lon, 10.0)),
    }
    reg["locations"].append(entry)
    REGISTRY.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    terrain.close_all()
    print(f"  registered {gid}  {entry['name']}  tier={tier}"
          f"  susc_near={entry['susceptibility_nearby_max']}"
          f"  flood={'yes' if flood_ok else 'no'}"
          f"  people10km={entry['people_10km']:,}")
    return entry


SEED = [
    ("Dhaka, Bangladesh", 23.80, 90.40),
    ("Kathmandu, Nepal", 27.72, 85.32),
    ("Freetown, Sierra Leone", 8.48, -13.23),
    ("Seattle, USA", 47.61, -122.33),
    ("Portland West Hills, USA", 45.52, -122.72),
    ("Medellin, Colombia", 6.25, -75.56),
    ("La Paz, Bolivia", -16.49, -68.13),
    ("Chongqing, China", 29.56, 106.55),
    ("Blantyre, Malawi", -15.79, 35.00),
    ("Can Tho, Vietnam", 10.03, 105.78),
]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("lat", type=float)
    a.add_argument("lon", type=float)
    a.add_argument("--name")
    sub.add_parser("seed")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "add":
        register(args.lat, args.lon, args.name)
    elif args.cmd == "seed":
        for name, la, lo in SEED:
            register(la, lo, name)
    else:
        for x in _load()["locations"]:
            print(f"  {x['id']}  {x['name']:<28} tier {x['tier_susceptibility']}"
                  f"  flood={'y' if x['flood_basin_cached'] else 'n'}")
