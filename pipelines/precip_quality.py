"""Precipitation source-agreement check: detect faulty data before trusting it.

Born from a real failure: during the 2022 Horn of Africa famine drought, NASA
POWER's PRECTOTCORR at (2.0N, 45.0E) reports 2022 as the WETTEST year in its
record (1,192 mm) with heavy rain in the bone-dry Hagaa season -- while ERA5
reports 557 mm and correctly shows the drought breaking in 2023. Ground truth
(FEWS NET) sides with ERA5. One reanalysis is unusable in that region, and a
percentile computed on unusable data is confidently wrong.

The defense is disagreement detection: fetch a second, independent source
(ERA5 via the Open-Meteo archive) for a short comparison window and score
agreement on monthly totals. Where the sources diverge, no precip-derived
product (SPI, rain trigger) deserves Tier B -- degrade to Tier C and say why.
This is the user's "warn even with possibly faulty info" requirement made
mechanical: the system knows when its own inputs are untrustworthy.

Agreement is judged on 8 years of monthly totals:
  corr      Pearson correlation of monthly series   (seasonality + events)
  ratio     mean annual POWER / mean annual ERA5    (systematic bias)
Verdict "ok" needs corr >= 0.60 and ratio in [0.6, 1.6]; else "disagree".

Cached per cell; one small ERA5 request per new cell (~8 years x 1 variable
stays friendly to the free tier). A production deployment would precompute a
global agreement mask once and refresh it yearly.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402
from pipelines import nasapower  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

PQ_DIR = CACHE / "precip_quality"
PQ_DIR.mkdir(parents=True, exist_ok=True)
ERA5 = "https://archive-api.open-meteo.com/v1/archive"
CMP_START, CMP_END = "2016-01-01", "2023-12-31"


def _monthly(times, vals) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for d, v in zip(times, vals):
        out[d[:7]] += (v or 0.0)
    return dict(out)


def check(lat: float, lon: float) -> dict:
    """Agreement verdict for the POWER precip cell containing (lat, lon)."""
    c = nasapower.cell(lat, lon)
    cache = PQ_DIR / f"{c[0]:+06.1f}_{c[1]:+07.1f}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    raw = nasapower.fetch_cell(*c)
    if raw is None:
        return {"verdict": "no_data", "detail": "no POWER series here"}
    mp = _monthly(raw["time"], raw["precipitation_sum"])

    try:
        r = SESSION.get(ERA5, params={
            "latitude": c[0], "longitude": c[1],
            "start_date": CMP_START, "end_date": CMP_END,
            "daily": "precipitation_sum", "timezone": "UTC"}, timeout=120)
        if r.status_code != 200:
            return {"verdict": "unverified",
                    "detail": f"ERA5 unavailable (HTTP {r.status_code}); "
                              "agreement not checkable right now"}
        d = r.json()["daily"]
        me = _monthly(d["time"], d["precipitation_sum"])
    except Exception as e:                                  # noqa: BLE001
        return {"verdict": "unverified", "detail": f"ERA5 fetch failed: {e}"}

    months = sorted(set(mp) & set(me))
    months = [m for m in months if CMP_START[:7] <= m <= CMP_END[:7]]
    a = np.array([mp[m] for m in months])
    b = np.array([me[m] for m in months])
    if len(months) < 48 or b.sum() <= 0:
        return {"verdict": "unverified", "detail": "insufficient overlap"}
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    ratio = float(a.sum() / b.sum())
    verdict = "ok" if (corr >= 0.60 and 0.6 <= ratio <= 1.6) else "disagree"
    out = {"verdict": verdict, "corr_monthly": round(corr, 3),
           "annual_ratio_power_over_era5": round(ratio, 3),
           "n_months": len(months),
           "detail": ("POWER and ERA5 agree here" if verdict == "ok" else
                      "POWER and ERA5 tell different stories at this cell -- "
                      "precip-derived scores are not trustworthy")}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    for name, la, lo in [("horn-of-africa", 2.0, 45.0),
                         ("cape-town", -33.92, 18.42),
                         ("iowa", 41.9, -93.6),
                         ("pnw-coast-range", 45.5, -123.5)]:
        v = check(la, lo)
        print(f"{name:<18} {v['verdict']:<10} "
              f"corr={v.get('corr_monthly')}  ratio={v.get('annual_ratio_power_over_era5')}")
