"""Batch scoring: combine the two layers into one public-facing status.

v1 uses the LHASA-style decision rule the brief asks for -- both layers must
cross their own threshold -- rather than a jointly trained model. It is easier
to debug, easier to explain to someone acting on it, and it keeps the two
signals legible: "this slope is vulnerable" and "today is unusual here" are
different statements and a user is entitled to see which one fired.

Thresholds come from each model's recall-tuned operating point (recorded in
models/runs/*.json), not from round numbers. The brief tunes for recall over
precision: missing a real landslide is worse than a false alarm.

Writes one flat JSON per location for the static site to fetch. No live
per-request compute.
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
from pipelines import nasapower, openmeteo  # noqa: E402

WEATHER = {"nasapower": nasapower, "openmeteo": openmeteo}


def weather_source():
    """Score with the SAME rainfall source the trigger model was trained on.

    Mixing them would silently feed the model percentiles computed from a
    different product on a different grid -- the features would still be
    numerically valid, and the predictions quietly wrong.
    """
    man = PROCESSED / "features_manifest.json"
    if man.exists():
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            src = (m.get("layers", {}).get("trigger", {}) or {}).get("weather_source")
            if src in WEATHER:
                return WEATHER[src], src
        except json.JSONDecodeError:
            pass
    return nasapower, "nasapower"

ARTIFACTS = ROOT / "models" / "artifacts"
RUNS = ROOT / "models" / "runs"
OUT = ROOT / "serve" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Fallback operating points if a run has no recall-tuned threshold recorded.
DEFAULT_THRESH = {"susceptibility": 0.5, "trigger": 0.5}


def load_layer(layer: str):
    p = ARTIFACTS / f"{layer}.pkl"
    if not p.exists():
        return None
    with p.open("rb") as fh:
        return pickle.load(fh)


def operating_point(layer: str) -> float:
    """Recall-tuned threshold from the most recent run for this layer."""
    best, best_time = None, ""
    for f in RUNS.glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if r.get("layer") != layer or r.get("status") != "complete":
            continue
        if r.get("trained_at", "") >= best_time:
            best, best_time = r, r.get("trained_at", "")
    if not best:
        return DEFAULT_THRESH[layer]
    rt = (best.get("summary") or {}).get("recall_threshold")
    return float(rt["threshold"]) if rt else DEFAULT_THRESH[layer]


def predict(bundle, feats: dict) -> float | None:
    """Calibrated probability from a layer bundle, or None on missing inputs."""
    if bundle is None:
        return None
    names = bundle["features"]
    x = np.array([[feats.get(n, np.nan) for n in names]], dtype="float64")
    if np.isnan(x).all():
        return None
    p = float(bundle["model"].predict_proba(x)[:, 1][0])
    cal = bundle.get("calibrator")
    return float(cal.predict([p])[0]) if cal is not None else p


SENTENCES = {
    "red": ("Rain here is well above normal for this time of year, and this is the "
            "kind of slope that fails. Take care on and below steep ground today."),
    "amber": ("Rain here is above normal for this time of year. The ground here is "
              "less prone to sliding, but stay aware."),
    "green": ("Rainfall here is normal for this time of year. Nothing unusual today."),
    "unknown": ("Not enough data to score this location yet."),
}


def status_for(susc: float | None, trig: float | None,
               s_thr: float, t_thr: float) -> tuple[str, str]:
    """Combine the two layers, gated on the TIME-VARYING one.

    The obvious rule -- "amber if either layer crosses" -- is wrong for a
    warning product. Susceptibility is a property of the hillside: it does not
    change day to day, so a location on steep ground would sit at amber every
    single day of the year, including a dry July afternoon. A permanent warning
    is not a warning, and it trains people to ignore the thing.

    So the trigger gates the alert and susceptibility sets its severity, which
    is also how LHASA reports hazard. Susceptibility is still published in the
    payload as standing context, it just does not raise an alarm on its own.
    """
    if susc is None or trig is None:
        return "unknown", SENTENCES["unknown"]
    if trig < t_thr:
        return "green", SENTENCES["green"]
    return ("red", SENTENCES["red"]) if susc >= s_thr else ("amber", SENTENCES["amber"])


def score_location(loc_id: str, lat: float, lon: float, date: str,
                   susc_b, trig_b, s_thr: float, t_thr: float, wx=None) -> dict:
    t = terrain.derive(lat, lon) if susc_b else None
    # Don't touch the weather API at all if there is no trigger model to feed --
    # a partial deployment should degrade, not do pointless network work.
    w = wx.features_at(lat, lon, date) if (trig_b and wx) else None
    susc = predict(susc_b, t) if t else None
    trig = predict(trig_b, w) if w else None
    state, sentence = status_for(susc, trig, s_thr, t_thr)
    return {
        "location_id": loc_id,
        "lat": round(lat, 5), "lon": round(lon, 5),
        "date": date,
        "status": state,
        "message": sentence,
        "susceptibility": None if susc is None else round(susc, 4),
        "trigger": None if trig is None else round(trig, 4),
        "thresholds": {"susceptibility": round(s_thr, 4), "trigger": round(t_thr, 4)},
        "drivers": None if not w else {
            "precip_3d_mm": round(w.get("precip_3d", float("nan")), 1),
            "precip_3d_pctl_seasonal": round(w.get("precip_3d_pctl_seasonal", float("nan")), 3),
            "precip_30d_over_normal": round(w.get("precip_30d_over_climo_mean", float("nan")), 2),
        },
        "terrain": None if not t else {
            "slope_deg": round(t["slope_deg"], 1),
            "relief_m": round(t["relief_range"], 0),
            "elevation_m": round(t["elev_m"], 0),
        },
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "caveat": ("Relative risk within this study region, not an absolute "
                   "probability. Not an official warning; consult local "
                   "authorities."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="scoring date (default: last available)")
    ap.add_argument("--locations", default=None,
                    help="JSON file of [{id,lat,lon}]; defaults to a demo set")
    a = ap.parse_args()

    wx, wx_name = weather_source()
    susc_b, trig_b = load_layer("susceptibility"), load_layer("trigger")
    if susc_b is None and trig_b is None:
        raise SystemExit("no trained models in models/artifacts/. Run models/train.py first.")
    s_thr, t_thr = operating_point("susceptibility"), operating_point("trigger")
    print(f"weather source: {wx_name}")
    print(f"operating points: susceptibility>={s_thr:.3f}  trigger>={t_thr:.3f}")

    date = a.date or wx.END
    if a.locations:
        locs = json.loads(Path(a.locations).read_text(encoding="utf-8"))
    else:
        locs = [
            {"id": "or-coast-range", "lat": 45.50, "lon": -123.50},
            {"id": "portland-west-hills", "lat": 45.52, "lon": -122.72},
            {"id": "columbia-gorge", "lat": 45.60, "lon": -121.95},
            {"id": "willamette-floor", "lat": 45.20, "lon": -123.10},
            {"id": "olympic-foothills", "lat": 47.60, "lon": -123.40},
        ]

    index = []
    for L in locs:
        rec = score_location(L["id"], L["lat"], L["lon"], date,
                             susc_b, trig_b, s_thr, t_thr, wx)
        (OUT / f"{L['id']}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        index.append({k: rec[k] for k in ("location_id", "lat", "lon", "status",
                                          "susceptibility", "trigger")})
        print(f"  {L['id']:<22} {rec['status']:<8} "
              f"susc={rec['susceptibility']}  trig={rec['trigger']}")
    (OUT / "index.json").write_text(
        json.dumps({"date": date, "locations": index,
                    "generated_at": dt.datetime.now().isoformat(timespec="seconds")},
                   indent=2), encoding="utf-8")
    terrain.close_all()
    print(f"\nwrote {len(index)+1} files -> {OUT}")


if __name__ == "__main__":
    main()
