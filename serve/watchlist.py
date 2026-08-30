"""The standing watchlist: 50 highly-exposed cities, every hazard, one date.

This is the platform's heartbeat made concrete. For a scored world date
(serve/score_world.py output), read each city's weather-hazard percentiles
straight from the world fields (instant), attach river state where a GloFAS
basin is cached, landslide susceptibility where terrain is reachable, and
people_near() so every row says who it is about. Output: one JSON the static
site can serve verbatim, plus a ranked console summary.

City list: flood/landslide/fire/drought-exposed metros chosen for hazard and
regime spread, not importance — the point is that the machinery answers
everywhere, tiers and caveats included.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, ROOT  # noqa: E402

CITIES = [
    # name, lat, lon  (spread across regimes and hazards)
    ("Dhaka, Bangladesh", 23.80, 90.40),
    ("Sylhet, Bangladesh", 24.90, 91.87),
    ("Karachi, Pakistan", 24.86, 67.01),
    ("Sukkur, Pakistan", 27.70, 68.87),
    ("Mumbai, India", 19.08, 72.88),
    ("Kolkata, India", 22.57, 88.36),
    ("Kathmandu, Nepal", 27.72, 85.32),
    ("Guwahati, India", 26.14, 91.74),
    ("Yangon, Myanmar", 16.87, 96.20),
    ("Hanoi, Vietnam", 21.03, 105.85),
    ("Can Tho, Vietnam", 10.03, 105.78),
    ("Phnom Penh, Cambodia", 11.56, 104.92),
    ("Bangkok, Thailand", 13.76, 100.50),
    ("Vientiane, Laos", 17.97, 102.60),
    ("Manila, Philippines", 14.60, 120.98),
    ("Jakarta, Indonesia", -6.21, 106.85),
    ("Wuhan, China", 30.59, 114.31),
    ("Chongqing, China", 29.56, 106.55),
    ("Lagos, Nigeria", 6.52, 3.38),
    ("Niamey, Niger", 13.51, 2.13),
    ("Khartoum, Sudan", 15.50, 32.56),
    ("Mogadishu, Somalia", 2.05, 45.32),
    ("Nairobi, Kenya", -1.29, 36.82),
    ("Addis Ababa, Ethiopia", 9.02, 38.75),
    ("Freetown, Sierra Leone", 8.48, -13.23),
    ("Kinshasa, DR Congo", -4.44, 15.27),
    ("Blantyre, Malawi", -15.79, 35.00),
    ("Cape Town, South Africa", -33.92, 18.42),
    ("Cairo, Egypt", 30.04, 31.24),
    ("Istanbul, Turkiye", 41.01, 28.98),
    ("Tehran, Iran", 35.69, 51.39),
    ("Baghdad, Iraq", 33.31, 44.36),
    ("Cologne, Germany", 50.94, 6.96),
    ("Valencia, Spain", 39.47, -0.38),
    ("Lisbon, Portugal", 38.72, -9.14),
    ("Athens, Greece", 37.98, 23.73),
    ("Rio de Janeiro, Brazil", -22.91, -43.17),
    ("Sao Paulo, Brazil", -23.55, -46.63),
    ("Porto Alegre, Brazil", -30.03, -51.23),
    ("La Paz, Bolivia", -16.49, -68.13),
    ("Medellin, Colombia", 6.25, -75.56),
    ("Guatemala City, Guatemala", 14.63, -90.55),
    ("Port-au-Prince, Haiti", 18.54, -72.34),
    ("Mexico City, Mexico", 19.43, -99.13),
    ("Los Angeles, USA", 34.05, -118.24),
    ("Phoenix, USA", 33.45, -112.07),
    ("New Orleans, USA", 29.95, -90.07),
    ("Seattle, USA", 47.61, -122.33),
    ("Sydney, Australia", -33.87, 151.21),
    ("Athens-of-the-North Edinburgh, UK", 55.95, -3.19),
]

SIGNALS = ["rain3d", "rain30d", "spi90", "spi180", "spi365", "kbdi", "vpd"]


def main(date: str) -> None:
    src = PROCESSED / "world" / date
    if not src.exists():
        raise SystemExit(f"score the world first: serve/score_world.py --date {date}")
    z = np.load(PROCESSED / "global_ladders.npz")
    lat, lon = z["lat"], z["lon"]
    fields = {s: np.load(src / f"{s}_pctl.npy").astype("float32")
              for s in SIGNALS}

    from pipelines.population import people_near
    from serve.score_global import flood_block

    wp = PROCESSED / "world" / f"window_{date}.npz"
    window_P = np.load(wp)["P"] if wp.exists() else None

    valid = np.isfinite(fields["rain3d"])      # a real land climate cell

    def nearest_valid(la, lo):
        """Coastal metros (Freetown's peninsula, Manila bay) can sit in an
        ocean-degenerate 0.5-deg cell: rain ladder NaN, KBDI pinned. Snap to
        the nearest VALID cell within +/-2 -- same move as the river-channel
        snap, for the same reason."""
        i = int(np.argmin(np.abs(lat - la)))
        j = int(np.argmin(np.abs(lon - lo)))
        if valid[i, j]:
            return i, j, False
        best = None
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ii, jj = i + di, j + dj
                if 0 <= ii < len(lat) and 0 <= jj < len(lon) and valid[ii, jj]:
                    d2 = di * di + dj * dj
                    if best is None or d2 < best[0]:
                        best = (d2, ii, jj)
        return (best[1], best[2], True) if best else (i, j, False)

    rows = []
    for name, la, lo in CITIES:
        i, j, snapped = nearest_valid(la, lo)
        r = {"city": name, "lat": la, "lon": lo,
             "people_10km": int(people_near(la, lo, 10.0)),
             "cell_snapped_to_land": snapped}
        for s in SIGNALS:
            v = float(fields[s][i, j])
            r[s] = None if not np.isfinite(v) else round(v, 3)
        fb = flood_block(la, lo, date)
        r["flood"] = {k: fb.get(k) for k in
                      ("tier", "is_river", "flow_pctl_seasonal", "alert")}
        flags = []
        ok = r.get("rain3d") is not None            # flags only on valid climate
        if ok and ((r["rain3d"] >= 0.98) or ((r.get("rain30d") or 0) >= 0.98)):
            flags.append("extreme_rain")
        if ok and (r.get("vpd") or 0) >= 0.95 and (r.get("kbdi") or 0) >= 0.90:
            flags.append("fire_weather")
        if ok and r.get("spi90") is not None and r["spi90"] <= 0.05:
            flags.append("drought_3mo")
        if ok and r.get("spi365") is not None and r["spi365"] <= 0.05:
            flags.append("drought_structural")
        if fb.get("alert"):
            flags.append("river_high")

        # Verify-on-alert (D24's detector, applied before warning anyone):
        # a precip-derived flag only stands if POWER and ERA5 agree at this
        # cell. Freetown forced this: the archive claims a near-record-dry
        # Jun-Jul 2026 on a coast where reanalyses are known to struggle.
        precip_flags = {"extreme_rain", "drought_3mo", "drought_structural",
                        "fire_weather"}
        if any(f in precip_flags for f in flags):
            try:
                from pipelines.precip_quality import check, recent_vs_era5
                pq = check(la, lo)
                # freshness: POWER side straight from the scored zarr window
                if window_P is not None:
                    pmm = float(np.nansum(window_P[-120:, i, j]))
                    rec = recent_vs_era5(la, lo, date, pmm)
                    pq = {**pq, **rec}
                    if rec.get("recent") == "disagree":
                        pq["verdict"] = "disagree"
            except Exception:                               # noqa: BLE001
                pq = {"verdict": "unverified"}
            r["precip_sources"] = pq.get("verdict")
            if pq.get("verdict") == "disagree":
                r["unverified_flags"] = [f for f in flags if f in precip_flags]
                flags = [f for f in flags if f not in precip_flags]
                r["data_quality"] = (
                    "precipitation sources disagree at this cell "
                    f"(corr {pq.get('corr_monthly')}, ratio "
                    f"{pq.get('annual_ratio_power_over_era5')}) -- "
                    "flags withheld, treat as unknown")
        r["flags"] = flags
        rows.append(r)

    rows.sort(key=lambda r: (-len(r["flags"]), -(r.get("rain30d") or 0)))
    out = {"date": date,
           "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
           "n_cities": len(rows), "cities": rows,
           "note": ("weather percentiles vs each place's own 2001+ climatology; "
                    "flood where a GloFAS basin is cached; not an official warning")}
    p = ROOT / "serve" / "out_global" / f"watchlist_{date}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{'city':<28}{'rain30d':>8}{'spi90':>7}{'kbdi':>6}{'vpd':>6}"
          f"{'river':>7}  flags")
    def fv(x, w):
        return f"{x:>{w}.2f}" if isinstance(x, (int, float)) else " " * (w - 3) + " --"
    for r in rows[:18]:
        fl = r["flood"]["flow_pctl_seasonal"]
        print(f"{r['city'][:27]:<28}{fv(r.get('rain30d'),8)}{fv(r.get('spi90'),7)}"
              f"{fv(r.get('kbdi'),6)}{fv(r.get('vpd'),6)}{fv(fl,7)}  "
              f"{','.join(r['flags']) or '-'}"
              + ("  [snapped]" if r.get("cell_snapped_to_land") else "")
              + ("  [SOURCES DISAGREE: " + ",".join(r.get("unverified_flags", []))
                 + " withheld]" if r.get("data_quality") else ""))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    main(ap.parse_args().date)
