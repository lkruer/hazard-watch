"""Nightly update: fresh weather for every registered location.

The cron's whole job. For each entry in serve/locations.json (statics were
computed at registration): pull the fresh weather tail (~3-day lag, POWER
point API), compute all weather-driven hazard scores on the newest scorable
day, verify the tail against ERA5 before trusting any alert (D31), compose
the one-color one-sentence status the brief specifies, write
serve/out_live/f/<id>.json, and append the day to that location's history
(the site's trend chart).

Light by design: no DEM, no ladders, no big files -- runs on a laptop cron or
a free CI runner identically.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT  # noqa: E402
from eval.drought_validate import FEATURES as DR_FEATS  # noqa: E402
from eval.drought_validate import DroughtSeries  # noqa: E402
from pipelines import fresh  # noqa: E402
from pipelines import fireweather as fw  # noqa: E402
from pipelines.precip_quality import check, recent_vs_era5  # noqa: E402

OUT = ROOT / "serve" / "out_live"
HIST = OUT / "history"
THRESH = ROOT / "serve" / "thresholds.json"
HISTORY_KEEP = 180


def _load_pickle(name):
    import pickle
    p = ROOT / "models" / "artifacts" / name
    if not p.exists():
        return None
    with p.open("rb") as fh:
        return pickle.load(fh)


def _pred(bundle, feats: dict) -> float | None:
    if bundle is None or feats is None:
        return None
    x = np.array([[feats.get(k, np.nan) for k in bundle["features"]]])
    if np.isnan(x).all():
        return None
    p = float(bundle["model"].predict_proba(x)[:, 1][0])
    cal = bundle.get("calibrator")
    return float(cal.predict([p])[0]) if cal is not None else p


def compose(loc: dict, w: dict) -> tuple[str, str]:
    """One dominant color + one plain sentence. Acute hazards outrank slow
    ones; data problems outrank silence."""
    flood_p = w.get("flood_pctl")
    # The river signal comes from GloFAS's own meteorology, not from the
    # POWER series the quality gate audits -- so a flood red stands even
    # where rain sources disagree. Khartoum forced this: the Nile at its
    # seasonal record (1.00) was being served as "unknown".
    if flood_p is not None and flood_p >= 0.98:
        return "red", ("The river is exceptionally high for this time of "
                       "year. Stay aware of water levels and local warnings.")
    if w.get("data_quality") == "disagree":
        extra = ""
        if flood_p is not None and flood_p >= 0.95:
            extra = (" The river, measured independently, IS running high -- "
                     "watch water levels.")
        return "unknown", ("Rainfall sources disagree here right now, so "
                           "rain-based scores are not trustworthy. Treat "
                           "those as unknown and rely on local guidance."
                           + extra)
    rain = max(w.get("rain3d_pctl") or 0, w.get("rain30d_pctl") or 0)
    susc = loc.get("susceptibility_nearby_max") or 0
    fire_a = w.get("fire_alert")

    if rain >= 0.98 and susc >= 0.30:
        return "red", ("Rain here is extreme for this time of year and the "
                       "slopes nearby are the kind that fail. Be careful on "
                       "and below steep ground.")
    if fire_a:
        return "red", ("Fire weather is dangerous today: unusually hot, dry "
                       "conditions. A fire that starts can spread fast.")
    yellow = []
    if rain >= 0.95:
        yellow.append("rain is unusually heavy for the season")
    if flood_p is not None and flood_p >= 0.95:
        yellow.append("the river is running high")
    if w.get("fire_watch"):
        yellow.append("fire weather is elevated")
    if (w.get("spi90") or 1) <= 0.05:
        yellow.append("the last three months have been exceptionally dry")
    if (w.get("spi365") or 1) <= 0.05:
        yellow.append("a long drought is in progress")
    wk = w.get("flood_week_max_pctl")
    if wk is not None and wk >= 0.98 and (flood_p or 0) < 0.98:
        yellow.append("the river is forecast to run exceptionally high "
                      f"around {w.get('flood_week_max_date')}")
    if yellow:
        return "yellow", ("Worth watching: " + "; ".join(yellow) +
                          ". Not an alert, but conditions are unusual.")
    if w.get("rain3d_pctl") is None:
        return "unknown", "No fresh weather available for this location."
    return "green", "Nothing unusual for this location today."


def update_one(loc: dict, trig_b, fire_b, drought_b, fire_thr, fire_watch_thr,
               glofas_stack=None) -> dict:
    la, lo = loc["lat"], loc["lon"]
    w: dict = {}

    rs = fresh.rain_series(la, lo)
    date = fresh.last_valid_date(rs) if rs else None
    if rs is not None and date:
        f = rs.features(date)
        if f:
            w["rain3d_pctl"] = round(f["precip_3d_pctl_seasonal"], 3)
            w["rain30d_pctl"] = round(f["precip_30d_pctl_seasonal"], 3)
            w["rain3d_mm"] = round(f["precip_3d"], 1)
        ds = DroughtSeries({"time": list(rs.idx),
                            "precipitation_sum": [None if not np.isfinite(v)
                                                  else float(v)
                                                  for v in rs.precip]})
        dfeat = ds.features(date)
        if dfeat:
            w["spi90"] = round(dfeat["spi90"], 3)
            w["spi365"] = round(dfeat["spi365"], 3)
            w["p_severe_drought"] = (None if drought_b is None else
                                     round(_pred(drought_b, dfeat) or 0, 3))
        # D31 freshness gate: verify the tail before trusting anything acute
        pmm = float(np.nansum(rs.precip[-120:]))
        longrun = check(la, lo)
        rec = recent_vs_era5(la, lo, date, pmm)
        from pipelines.precip_quality import combine_verdict
        cv = combine_verdict(longrun, rec)
        if cv["verdict"] == "disagree":
            w["data_quality"] = "disagree"
            w["data_quality_reasons"] = cv["reasons"]
            w["data_quality_detail"] = {**{k: longrun.get(k) for k in
                                           ("corr_monthly",
                                            "annual_ratio_power_over_era5")},
                                        **rec}

    fs = fresh.fire_series(la, lo)
    fdate = fresh.last_valid_date(fs, "tmax") if fs else None
    if fs is not None and fdate:
        ff = fs.features(fdate)
        if ff:
            danger = _pred(fire_b, ff)
            w["fire_danger"] = None if danger is None else round(danger, 3)
            w["fire_alert"] = bool(danger is not None and danger >= fire_thr
                                   and w.get("data_quality") != "disagree")
            w["fire_watch"] = bool(danger is not None
                                   and danger >= fire_watch_thr)
            w["kbdi"] = round(ff["kbdi"], 0)
            w["fire_date"] = fdate

    if glofas_stack is not None and date:
        try:
            r = glofas_stack.percentile_at(la, lo, date)
        except Exception:                                   # noqa: BLE001
            r = None
        if r and r.get("is_river") and date <= "2024-12-31":
            w["flood_pctl"] = r["flow_pctl_seasonal"]
            w["flood_m3s"] = r["discharge_m3s"]
        elif date > "2024-12-31":
            # live dates ride the operational FORECAST (D33): control run,
            # ranked against the same historical channel cell's seasonal record
            try:
                from pipelines.glofas_forecast import outlook
                fc = outlook(la, lo, glofas_stack)
            except Exception:                               # noqa: BLE001
                fc = None
            if fc and fc.get("is_river"):
                w["flood_pctl"] = fc["today"]["flow_pctl_seasonal"]
                w["flood_m3s"] = fc["today"]["discharge_m3s"]
                w["flood_week_max_pctl"] = fc["week_max"]["flow_pctl_seasonal"]
                w["flood_week_max_date"] = fc["week_max"]["valid"]
                w["flood_source"] = f"GloFAS operational forecast (init {fc['init']})"
            else:
                w["flood_note"] = "no river forecast fetchable right now"

    color, sentence = compose(loc, w)
    rec_out = {
        "location_id": loc["id"], "name": loc["name"],
        "lat": la, "lon": lo,
        "status": color, "message": sentence,
        "as_of": date, "generated_at":
            dt.datetime.now().isoformat(timespec="seconds"),
        "weather": w,
        "static": {k: loc.get(k) for k in
                   ("tier_susceptibility", "region_model", "susceptibility",
                    "susceptibility_nearby_max", "terrain", "people_10km")},
        "caveats": ["percentiles vs this location's own 2004-present record",
                    "not an official warning; consult local authorities"],
    }
    (OUT / "f").mkdir(parents=True, exist_ok=True)
    (OUT / "f" / f"{loc['id']}.json").write_text(
        json.dumps(rec_out, indent=1), encoding="utf-8")

    HIST.mkdir(parents=True, exist_ok=True)
    hp = HIST / f"{loc['id']}.jsonl"
    lines = hp.read_text(encoding="utf-8").splitlines() if hp.exists() else []
    lines = [x for x in lines if f'"as_of": "{date}"' not in x]
    lines.append(json.dumps({
        "as_of": date, "status": color,
        "rain3d_pctl": w.get("rain3d_pctl"), "rain30d_pctl": w.get("rain30d_pctl"),
        "spi90": w.get("spi90"), "fire_danger": w.get("fire_danger"),
        "flood_pctl": w.get("flood_pctl")}))
    hp.write_text("\n".join(lines[-HISTORY_KEEP:]) + "\n", encoding="utf-8")
    return rec_out


def main() -> None:
    reg = json.loads((ROOT / "serve" / "locations.json").read_text("utf-8"))
    thr = json.loads(THRESH.read_text("utf-8"))
    fire_thr = float(thr["fire"]["threshold"])
    fire_watch_thr = float(thr["fire"]["budgets"]["0.10"]["threshold"])
    trig_b = _load_pickle("trigger.pkl")
    fire_b = _load_pickle("fire_trigger.pkl")
    drought_b = _load_pickle("drought_head.pkl")

    # one shared discharge stack per cached basin, loaded lazily
    stacks: dict = {}

    def stack_of(loc):
        if not loc.get("flood_basin_cached"):
            return None
        from pipelines.glofas import stack_for
        key = (round(loc["lat"], 0), round(loc["lon"], 0))
        if key not in stacks:
            try:
                stacks[key] = stack_for(loc["lat"], loc["lon"])[0]
            except Exception:                               # noqa: BLE001
                stacks[key] = None
        return stacks[key]

    print(f"updating {len(reg['locations'])} locations "
          f"({dt.datetime.now():%Y-%m-%d %H:%M})")
    index = []
    for loc in reg["locations"]:
        r = update_one(loc, trig_b, fire_b, drought_b,
                       fire_thr, fire_watch_thr, stack_of(loc))
        index.append({k: r[k] for k in ("location_id", "name", "status",
                                        "as_of")})
        print(f"  {r['status']:<8} {r['name']:<28} as of {r['as_of']}  "
              f"{r['message'][:58]}")
    (OUT / "index.json").write_text(json.dumps(
        {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
         "locations": index}, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
