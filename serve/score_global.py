"""Score ANY point on Earth for the predictable hazards, tiered by confidence.

This is the "proficient everywhere" contract in code. For a (lat, lon, date):

  landslide   susceptibility: Tier A regional model if the point falls in a
              modeled region (12 and counting), else the Tier B global floor --
              the D19 ensemble of slope, NASA global class, and the pooled
              multi-region model, each expressed as a percentile of the global
              training pool so they average on a common scale.
              trigger: rainfall percentile-vs-own-climatology. D14 showed the
              one-line rule carries the trained model's skill, and unlike the
              trained model it needs no labels, so it IS the global method.
  fire        danger-conditions model (KBDI + VPD + weather percentiles),
              validated on two continents and shown to TRANSFER across them
              (US->Canada ROC 0.77 vs local 0.81) -- Tier B globally with the
              stated calibration caveat, since every feature self-normalizes
              against local climatology.
  drought     empirical SPI percentiles (label-free, global), plus a P(severe
              drought) head calibrated against 20 years of US Drought Monitor.

Every hazard block carries "tier" and "caveats" -- the user's requirement that
places with weak or faulty info still get a warning, plus the honesty about
how much to trust it. Missing inputs degrade the block to tier C with the
reason named, never a silent wrong answer.

Usage:
    python serve/score_global.py --lat 27.98 --lon 86.92 --date 2024-07-15
    python serve/score_global.py --demo
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, ROOT  # noqa: E402
from features import terrain  # noqa: E402
from pipelines import fireweather as fw  # noqa: E402
from pipelines import nasapower  # noqa: E402
from eval.drought_validate import DroughtSeries  # noqa: E402

ART = ROOT / "models" / "artifacts"

# Modeled landslide regions: bbox derived from each region matrix at load time.
_REGION_CACHE: dict | None = None


def _load(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def _regions():
    global _REGION_CACHE
    if _REGION_CACHE is not None:
        return _REGION_CACHE
    import pandas as pd
    out = {}
    for p in sorted(PROCESSED.glob("region_*.csv")):
        name = p.stem.replace("region_", "")
        art = _load(ART / (f"susceptibility.pkl" if name == "pnw"
                           else f"susceptibility-{name}.pkl"))
        if art is None:
            continue
        df = pd.read_csv(p, usecols=["lat", "lon"])
        out[name] = {"bbox": (df.lon.min(), df.lat.min(),
                              df.lon.max(), df.lat.max()),
                     "artifact": art}
    _REGION_CACHE = out
    return out


def _pool_reference():
    """Global pool of slope values + pooled model, for Tier-B percentiles."""
    import pandas as pd
    frames = [pd.read_csv(p) for p in sorted(PROCESSED.glob("region_*.csv"))]
    pool = pd.concat(frames, ignore_index=True)
    from lightgbm import LGBMClassifier
    from models.train import BASE_PARAMS
    feats = [f for f in terrain.FEATURES if f in pool.columns]
    mdl = LGBMClassifier(**BASE_PARAMS).fit(
        pool[feats].to_numpy("float64"), pool["label"].to_numpy())
    ref_scores = np.sort(mdl.predict_proba(pool[feats].to_numpy("float64"))[:, 1])
    ref_slope = np.sort(pool["slope_deg"].to_numpy())
    return {"model": mdl, "features": feats,
            "ref_scores": ref_scores, "ref_slope": ref_slope}


_POOL = None
_PQ: dict = {}


def _precip_quality(lat: float, lon: float) -> dict:
    """Memoized source-agreement check (see pipelines/precip_quality.py)."""
    from pipelines import nasapower as _np
    from pipelines.precip_quality import check
    key = _np.cell(lat, lon)
    if key not in _PQ:
        _PQ[key] = check(lat, lon)
    return _PQ[key]


def _nasa_class(lat: float, lon: float) -> float | None:
    from pipelines.common import SESSION
    try:
        r = SESSION.post(
            "https://gis.earthdata.nasa.gov/gis01/rest/services/Landslides/"
            "Global_Landslide_Susceptibility/ImageServer/getSamples",
            data={"geometry": json.dumps({"points": [[lon, lat]],
                                          "spatialReference": {"wkid": 4326}}),
                  "geometryType": "esriGeometryMultipoint",
                  "returnFirstValueOnly": "true", "f": "json"}, timeout=30)
        v = float(str(r.json()["samples"][0]["value"]).split()[0])
        return v if 0 <= v <= 5 else None
    except Exception:                                       # noqa: BLE001
        return None


def _susc_from_terrain(t: dict, lat: float, lon: float) -> float | None:
    """Point susceptibility for a terrain dict, regional artifact preferred."""
    global _POOL
    hit = next(((n, r) for n, r in _regions().items()
                if r["bbox"][0] <= lon <= r["bbox"][2]
                and r["bbox"][1] <= lat <= r["bbox"][3]), None)
    if hit:
        b = hit[1]["artifact"]
        feats = {**t}
        if "road_dist_m" in b.get("features", ()):
            try:
                from pipelines.osm_roads import road_dist_m
                feats["road_dist_m"] = float(road_dist_m([lat], [lon])[0])
            except Exception:                               # noqa: BLE001
                feats["road_dist_m"] = float("nan")
        x = np.array([[feats.get(f, np.nan) for f in b["features"]]])
        pr = float(b["model"].predict_proba(x)[:, 1][0])
        cal = b.get("calibrator")
        return float(cal.predict([pr])[0]) if cal is not None else pr
    if _POOL is None:
        _POOL = _pool_reference()
    x = np.array([[t.get(f, np.nan) for f in _POOL["features"]]])
    pm = float(_POOL["model"].predict_proba(x)[:, 1][0])
    return float(np.searchsorted(_POOL["ref_scores"], pm)
                 / len(_POOL["ref_scores"]))


def landslide_block(lat: float, lon: float, date: str) -> dict:
    global _POOL
    t = terrain.derive(lat, lon)
    caveats = []
    if t is None:
        susc = None
        tier = "C"
        caveats.append("no DEM coverage at this point (ocean or missing tile)")
    else:
        hit = next(((n, r) for n, r in _regions().items()
                    if r["bbox"][0] <= lon <= r["bbox"][2]
                    and r["bbox"][1] <= lat <= r["bbox"][3]), None)
        if hit:
            name, reg = hit
            b = reg["artifact"]
            feats = {**t}
            if "road_dist_m" in b.get("features", ()):
                try:
                    from pipelines.osm_roads import road_dist_m
                    feats["road_dist_m"] = float(road_dist_m([lat], [lon])[0])
                except Exception:                           # noqa: BLE001
                    feats["road_dist_m"] = float("nan")
            x = np.array([[feats.get(f, np.nan) for f in b["features"]]])
            p = float(b["model"].predict_proba(x)[:, 1][0])
            cal = b.get("calibrator")
            susc = float(cal.predict([p])[0]) if cal is not None else p
            tier = "A"
            caveats.append(f"regional model: {name}")
        else:
            if _POOL is None:
                _POOL = _pool_reference()
            x = np.array([[t.get(f, np.nan) for f in _POOL["features"]]])
            p_model = float(_POOL["model"].predict_proba(x)[:, 1][0])
            pct_model = float(np.searchsorted(_POOL["ref_scores"], p_model)
                              / len(_POOL["ref_scores"]))
            pct_slope = float(np.searchsorted(_POOL["ref_slope"], t["slope_deg"])
                              / len(_POOL["ref_slope"]))
            parts = [pct_model, pct_slope]
            nc = _nasa_class(lat, lon)
            if nc is not None:
                parts.append(nc / 5.0)
            else:
                caveats.append("NASA global map unavailable here")
            susc = float(np.mean(parts))
            tier = "B"
            caveats.append("global-floor ensemble; relative score, not probability")

    # What looms NEARBY matters as much as the exact point: Oso's victims
    # lived on the flat valley floor 600 m from the scarp that killed them, and
    # the point score there reads 0.02. Score a ~900 m ring and keep the max.
    susc_near = susc
    if t is not None and susc is not None:
        vals = [susc]
        for dla in (-0.008, 0.0, 0.008):
            for dlo in (-0.008, 0.0, 0.008):
                if dla == 0.0 and dlo == 0.0:
                    continue
                tn = terrain.derive(lat + dla, lon + dlo)
                if tn is None:
                    continue
                pn = _susc_from_terrain(tn, lat + dla, lon + dlo)
                if pn is not None:
                    vals.append(pn)
        susc_near = float(max(vals))

    w = nasapower.features_at(lat, lon, date)
    if w is None:
        trig = None
        trig30 = None
        tier = "C"
        caveats.append("no rainfall record at this point/date")
    else:
        trig = float(w["precip_3d_pctl_seasonal"])
        # Antecedent saturation kills as surely as cloudbursts: Oso followed
        # 45 days of ~2x rain (30d pctl 0.96) with an unremarkable 3-day total
        # (0.33). Surface both; the alert gate takes the worse of the two.
        trig30 = float(w["precip_30d_pctl_seasonal"])
        pq = _precip_quality(lat, lon)
        if pq["verdict"] == "disagree":
            tier = "C"
            caveats.append("precipitation sources disagree at this cell "
                           f"(corr {pq.get('corr_monthly')}, ratio "
                           f"{pq.get('annual_ratio_power_over_era5')}) -- "
                           "rain trigger not trustworthy here")
    trig_worst = (max(x for x in (trig, trig30) if x is not None)
                  if (trig is not None or trig30 is not None) else None)
    return {"hazard": "landslide", "tier": tier,
            "susceptibility": None if susc is None else round(susc, 3),
            "susceptibility_nearby_max": (None if susc_near is None
                                          else round(susc_near, 3)),
            "trigger_rain_pctl_seasonal": None if trig is None else round(trig, 3),
            "trigger_rain30d_pctl_seasonal": (None if trig30 is None
                                              else round(trig30, 3)),
            "alert": (bool(trig_worst is not None and trig_worst >= 0.95
                           and susc_near is not None and susc_near >= 0.3)
                      if (trig_worst is not None and susc_near is not None)
                      else None),
            "caveats": caveats}


def fire_block(lat: float, lon: float, date: str) -> dict:
    b = _load(ART / "fire_trigger.pkl")
    f = fw.features_at(lat, lon, date)
    if f is None or b is None:
        return {"hazard": "fire", "tier": "C", "danger": None,
                "caveats": ["no fire-weather record at this point/date"]}
    x = np.array([[f.get(k, np.nan) for k in b["features"]]])
    p = float(b["model"].predict_proba(x)[:, 1][0])
    cal = b.get("calibrator")
    danger = float(cal.predict([p])[0]) if cal is not None else p
    # deployment-priced alarm threshold (D14's lesson applied to fire): the
    # score only becomes an alarm against the real distribution of days
    thr = None
    tj = ROOT / "serve" / "thresholds.json"
    if tj.exists():
        try:
            thr = float(json.loads(tj.read_text(encoding="utf-8"))
                        ["fire"]["threshold"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            thr = None
    out = {"hazard": "fire", "tier": "B",
           "danger": round(danger, 3),
           "alert": (bool(danger >= thr) if thr is not None else None),
           "kbdi": round(f["kbdi"], 0), "vpd_kpa": round(f["vpd_kpa"], 2),
           "kbdi_pctl_seasonal": round(f["kbdi_pctl_seasonal"], 3),
           "vpd_pctl_seasonal": round(f["vpd_pctl_seasonal"], 3),
           "tmax_pctl_seasonal": round(f["tmax_pctl_seasonal"], 3),
           "days_since_rain": int(f["days_since_rain"]),
           "caveats": ["danger conditions, not ignition prediction",
                       "calibrated on US+Canada fire records; validated to "
                       "transfer between them (ROC 0.77 cold)",
                       "0.5-deg daily wind cannot see downslope wind events"]}
    return out


def drought_block(lat: float, lon: float, date: str) -> dict:
    b = _load(ART / "drought_head.pkl")
    raw = nasapower.fetch_cell(*nasapower.cell(lat, lon))
    if raw is None:
        return {"hazard": "drought", "tier": "C", "spi90": None,
                "caveats": ["no precipitation record at this point"]}
    ds = DroughtSeries(raw)
    f = ds.features(date)
    if f is None:
        return {"hazard": "drought", "tier": "C", "spi90": None,
                "caveats": ["date outside computable window"]}
    out = {"hazard": "drought", "tier": "B",
           "spi30": round(f["spi30"], 3), "spi90": round(f["spi90"], 3),
           "spi180": round(f["spi180"], 3),
           "caveats": ["empirical SPI vs own 2004-2024 climatology"]}
    if "spi365" in f:
        out["spi365"] = round(f["spi365"], 3)
    pq = _precip_quality(lat, lon)
    if pq["verdict"] == "disagree":
        out["tier"] = "C"
        out["caveats"].append(
            "precipitation sources disagree at this cell (corr "
            f"{pq.get('corr_monthly')}, ratio "
            f"{pq.get('annual_ratio_power_over_era5')}) -- SPI values here "
            "are not trustworthy; treat as unknown")
    elif pq["verdict"] == "unverified":
        out["caveats"].append("source cross-check unavailable (rate limit); "
                              "single-source values")
    if b is not None:
        x = np.array([[f.get(k, np.nan) for k in b["features"]]])
        p = float(b["model"].predict_proba(x)[:, 1][0])
        cal = b.get("calibrator")
        out["p_severe_drought"] = round(
            float(cal.predict([p])[0]) if cal is not None else p, 3)
        out["caveats"].append("severity head calibrated on US Drought Monitor only")
    return out


def flood_block(lat: float, lon: float, date: str) -> dict:
    """GloFAS flow percentile at the nearest river cell -- integration, not
    modeling, per the brief. Tier B where a >=15-year record is cached;
    honest not-a-river / no-record answers otherwise."""
    try:
        from pipelines.glofas import stack_for
        stack, why = stack_for(lat, lon)
    except Exception as e:                                  # noqa: BLE001
        return {"hazard": "flood", "tier": "C", "flow_pctl_seasonal": None,
                "caveats": [f"discharge stack unavailable: {e}"]}
    if stack is None:
        return {"hazard": "flood", "tier": "C", "flow_pctl_seasonal": None,
                "caveats": [why]}
    r = stack.percentile_at(lat, lon, date)
    if r is None:
        return {"hazard": "flood", "tier": "C", "flow_pctl_seasonal": None,
                "caveats": ["date outside the cached discharge record"]}
    if not r.get("is_river"):
        return {"hazard": "flood", "tier": "B", "is_river": False,
                "flow_pctl_seasonal": None,
                "caveats": ["no river channel at this grid cell (GloFAS 0.05deg); "
                            "flood risk here is pluvial/flash, not riverine"]}
    return {"hazard": "flood", "tier": "B", "is_river": True,
            "flow_pctl_seasonal": r["flow_pctl_seasonal"],
            "discharge_m3s": r["discharge_m3s"],
            "median_here_m3s": r["median_here_m3s"],
            "alert": bool(r["flow_pctl_seasonal"] >= 0.98),
            "caveats": ["GloFAS v5 modeled discharge (Copernicus CEMS), "
                        "expressed against this cell's own 2004-2024 record; "
                        "validated against USGS gauges (see flood-validation)"]}


def score(lat: float, lon: float, date: str) -> dict:
    blocks = [landslide_block(lat, lon, date),
              fire_block(lat, lon, date),
              drought_block(lat, lon, date),
              flood_block(lat, lon, date)]
    hz = {b["hazard"]: b for b in blocks}
    # Compound-extremes summary: which signals sit in their local tail today.
    # Compound events (hot+dry+windy; saturated+steep) drive outsized losses,
    # and a flat list of tail signals reads at a glance.
    flags = []
    ls, fr, dr = hz["landslide"], hz["fire"], hz["drought"]
    if (ls.get("trigger_rain_pctl_seasonal") or 0) >= 0.98:
        flags.append("rainfall_3d_extreme_for_season")
    if (fr.get("kbdi_pctl_seasonal") or 0) >= 0.95:
        flags.append("fuel_dryness_extreme_for_season")
    if (fr.get("vpd_pctl_seasonal") or 0) >= 0.95:
        flags.append("atmospheric_dryness_extreme_for_season")
    if (fr.get("tmax_pctl_seasonal") or 0) >= 0.98:
        flags.append("heat_extreme_for_season")
    if dr.get("spi90") is not None and dr["spi90"] <= 0.05:
        flags.append("3-month_precipitation_deficit_severe")
    if dr.get("spi180") is not None and dr["spi180"] <= 0.05:
        flags.append("6-month_precipitation_deficit_severe")
    fl = hz.get("flood", {})
    if (fl.get("flow_pctl_seasonal") or 0) >= 0.98:
        flags.append("river_flow_extreme_for_season")
    return {"lat": lat, "lon": lon, "date": date,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "hazards": blocks,
            "compound_extremes": flags,
            "note": ("Free public-data hazard context. Not an official warning; "
                     "consult local authorities.")}


DEMO = [
    ("kathmandu-valley-rim", 27.75, 85.40, "2024-07-15"),   # monsoon Nepal (Tier A)
    ("swiss-alps-grindelwald", 46.62, 8.03, "2024-07-15"),  # unmodeled region (Tier B)
    ("paradise-california", 39.75, -121.60, "2018-11-08"),  # Camp Fire day
    ("okavango-botswana", -19.5, 23.0, "2019-10-15"),       # drought, unmodeled
    ("sahara-algeria", 27.0, 2.0, "2020-06-01"),            # null case
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--date", default="2024-06-15")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / "serve" / "out_global"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = (DEMO if a.demo or a.lat is None
             else [("point", a.lat, a.lon, a.date)])
    for name, la, lo, d in tasks:
        rec = score(la, lo, d)
        (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2),
                                              encoding="utf-8")
        hz = {h["hazard"]: h for h in rec["hazards"]}
        ls, fr, dr = hz["landslide"], hz["fire"], hz["drought"]
        print(f"\n{name}  ({la}, {lo})  {d}")
        print(f"  landslide[{ls['tier']}] susc={ls['susceptibility']} "
              f"rain3d_pctl={ls['trigger_rain_pctl_seasonal']}")
        print(f"  fire     [{fr['tier']}] danger={fr.get('danger')} "
              f"kbdi={fr.get('kbdi')} dsr={fr.get('days_since_rain')}")
        print(f"  drought  [{dr['tier']}] spi90={dr.get('spi90')} "
              f"p_severe={dr.get('p_severe_drought')}")
        fl = hz.get("flood", {})
        print(f"  flood    [{fl.get('tier')}] "
              f"flow_pctl={fl.get('flow_pctl_seasonal')} "
              f"q={fl.get('discharge_m3s')}")
    terrain.close_all()
    print(f"\nwrote {out_dir}")
