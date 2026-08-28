"""Score famous disasters through the global scorer: the credibility file.

Each case states the expectation BEFORE scoring, including where the system
should fail (wind-driven fires -- D20's documented gap). A validation story a
person can check against events they remember is worth as much as another AUC;
and every continent here exercises the full pipeline cold (DEM tile, POWER
weather, fire weather, drought series -- all fetched fresh for these points).

Honesty rule: the JSON records the expectation text verbatim next to the
scores, hits and misses alike.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT  # noqa: E402
from serve.score_global import score  # noqa: E402
from features import terrain  # noqa: E402

CASES = [
    {"name": "oso-landslide-WA", "lat": 48.2836, "lon": -121.8477,
     "date": "2014-03-22", "hazard": "landslide",
     "expect": ("Deadliest US landslide (43 dead). Inside the PNW Tier-A "
                "region after weeks of record March rain -- expect Tier A, "
                "elevated susceptibility, very high rain percentile.")},
    {"name": "freetown-mudslide-sierra-leone", "lat": 8.4260, "lon": -13.1910,
     "date": "2017-08-14", "hazard": "landslide",
     "expect": ("Regent/Freetown mudslide, ~1,100 dead, after 3 days of "
                "extreme rain in the wettest August stretch. Unmodeled region "
                "-> Tier B; the rain percentile should be extreme.")},
    {"name": "uttarakhand-kedarnath-india", "lat": 30.7346, "lon": 79.0669,
     "date": "2013-06-17", "hazard": "landslide",
     "expect": ("Himalayan flood/landslide disaster, >5,000 dead; early-"
                "monsoon deluge. Tier B; rain percentile should be near 1.0. "
                "Note: glacial-lake component is outside any rainfall model.")},
    {"name": "camp-fire-paradise-CA", "lat": 39.7596, "lon": -121.6219,
     "date": "2018-11-08", "hazard": "fire",
     "expect": ("Deadliest CA fire. Extreme fuel dryness after 200+ dry days "
                "-- KBDI should be extreme -- but it was a WIND event, and "
                "0.5-deg daily wind cannot see Jarbo Gap gusts. Expect high "
                "dryness, only moderate model danger: the documented gap.")},
    {"name": "fort-mcmurray-alberta", "lat": 56.7266, "lon": -111.3790,
     "date": "2016-05-03", "hazard": "fire",
     "expect": ("Costliest Canadian disaster. Record early-May heat (32C in "
                "boreal spring) after a dry El Nino winter. Expect seasonal "
                "temperature/VPD percentiles near 1.0 -- this one the weather "
                "features SHOULD catch.")},
    {"name": "black-saturday-victoria-AU", "lat": -37.4410, "lon": 145.5200,
     "date": "2009-02-07", "hazard": "fire",
     "expect": ("173 dead. Culmination of a record heatwave (46C in "
                "Melbourne) during the Millennium Drought. First Australian "
                "point ever scored -- full pipeline cold. Expect extreme "
                "KBDI/VPD/tmax percentiles.")},
    {"name": "marshall-fire-boulder-CO", "lat": 39.9550, "lon": -105.1660,
     "date": "2021-12-30", "hazard": "fire",
     "expect": ("Winter grass fire driven by 185 km/h chinook winds. "
                "Snowless dry autumn helps, but this is THE wind-gap case -- "
                "expect a documented miss on model danger.")},
    {"name": "iowa-drought-2012", "lat": 41.9, "lon": -93.6,
     "date": "2012-07-24", "hazard": "drought",
     "expect": ("Peak of the 2012 US flash drought, worst since the 1950s; "
                "USDM had most of Iowa in D3. Expect SPI-30/90 in the bottom "
                "few percent and a high severe-drought probability.")},
    {"name": "cape-town-day-zero", "lat": -33.9249, "lon": 18.4241,
     "date": "2018-02-01", "hazard": "drought",
     "expect": ("Day Zero crisis after three failed winter rain seasons. "
                "Expect SPI-180 (the long window) far below SPI-30 -- a "
                "structural, multi-season drought signature.")},
    {"name": "horn-of-africa-2022", "lat": 2.0, "lon": 45.0,
     "date": "2022-10-15", "hazard": "drought",
     "expect": ("Fifth consecutive failed rainy season in Somalia; famine "
                "conditions. Expect all three SPI windows depressed at once.")},
]


def main() -> None:
    out = []
    for c in CASES:
        print(f"\n=== {c['name']}  ({c['lat']}, {c['lon']})  {c['date']} ===")
        print(f"  expect: {c['expect'][:110]}...")
        try:
            rec = score(c["lat"], c["lon"], c["date"])
        except Exception as e:                              # noqa: BLE001
            print(f"  SCORING FAILED: {e}")
            out.append({**c, "result": None, "error": str(e)})
            continue
        hz = {h["hazard"]: h for h in rec["hazards"]}
        ls, fr, dr = hz["landslide"], hz["fire"], hz["drought"]
        print(f"  landslide[{ls['tier']}] susc={ls['susceptibility']} "
              f"rain3d_pctl={ls['trigger_rain_pctl_seasonal']}")
        print(f"  fire[{fr['tier']}] danger={fr.get('danger')} "
              f"kbdi={fr.get('kbdi')} vpd={fr.get('vpd_kpa')} "
              f"dsr={fr.get('days_since_rain')}")
        print(f"  drought[{dr['tier']}] spi30={dr.get('spi30')} "
              f"spi90={dr.get('spi90')} spi180={dr.get('spi180')} "
              f"p_severe={dr.get('p_severe_drought')}")
        out.append({**c, "result": rec})

    p = ROOT / "models" / "runs" / "case-studies.json"
    p.write_text(json.dumps({
        "name": "case-studies",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "note": ("expectations written before scoring; misses kept verbatim"),
        "cases": out}, indent=2, default=float), encoding="utf-8")
    terrain.close_all()
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
