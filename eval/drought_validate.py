"""Drought layer: empirical SPI from NASA POWER, validated against the USDM.

The drought indicator is the same trick that carried the landslide trigger and
the fire layer: a window total expressed as a percentile of the SAME calendar
window across all years at that location. That is precisely the empirical
Standardized Precipitation Index (SPI) -- McKee's index without the gamma-fit,
which matters at exactly zero of our decision points since only ranks are
used. spi30/90/180 = 1-, 3-, 6-month drought at any point on Earth POWER
covers, no labels needed.

Validation truth: the US Drought Monitor -- weekly expert-drawn national
drought maps since 2000, area-% per severity class per county, open API
(usdmdataservices.unl.edu, no key). We sample counties on a coarse grid across
CONUS so every climate regime votes: for each county, does low SPI at the
county centroid predict the experts marking the county >=50% D2 (severe
drought)?

Honest caveats stated up front: USDM is partly informed by the same
precipitation reality (not independent instrumentation), but it blends soil
moisture, streamflow, snowpack, groundwater and local expert judgement -- so
agreement is meaningful, and DISAGREEMENT tells us which droughts pure
precipitation misses (snowpack- and temperature-driven ones). The trained
"drought head" (SPI -> P(D2+)) is calibrated on US truth only; globally it
ships with that caveat.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, RAW, ROOT  # noqa: E402
from pipelines import nasapower  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

USDM = ("https://usdmdataservices.unl.edu/api/CountyStatistics/"
        "GetDroughtSeverityStatisticsByAreaPercent")
WINDOWS = (30, 90, 180, 365)   # 365: Cape-Town-class multi-season droughts
FEATURES = [f"spi{w}" for w in WINDOWS] + ["spi90_trend_8w"]
D2_THRESHOLD = 50.0          # county is "in severe drought" if >=50% area D2+
RUNS = ROOT / "models" / "runs"
CONUS_SKIP = {"AK", "HI", "PR", "GU", "VI", "MP", "AS"}


def pick_counties(step_deg: float = 4.0) -> pd.DataFrame:
    """One county per ~4-degree grid node across CONUS, from the Census
    gazetteer (public domain)."""
    with zipfile.ZipFile(RAW / "gaz_counties.zip") as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
        with z.open(name) as fh:
            g = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep="\t")
    g.columns = [c.strip() for c in g.columns]
    g = g[~g["USPS"].isin(CONUS_SKIP)].copy()
    g["GEOID"] = g["GEOID"].astype(str).str.zfill(5)
    lat, lon = g["INTPTLAT"].to_numpy(), g["INTPTLONG"].to_numpy()
    picked = []
    for glat in np.arange(26, 50, step_deg):
        for glon in np.arange(-124, -66, step_deg):
            d2 = (lat - glat) ** 2 + (lon - glon) ** 2
            i = int(np.argmin(d2))
            if d2[i] < (step_deg / 2) ** 2:
                picked.append(i)
    out = g.iloc[sorted(set(picked))][["USPS", "GEOID", "NAME",
                                       "INTPTLAT", "INTPTLONG"]]
    out = out.rename(columns={"INTPTLAT": "lat", "INTPTLONG": "lon"})
    print(f"picked {len(out)} counties across CONUS")
    return out.reset_index(drop=True)


def usdm_series(fips: str) -> pd.DataFrame | None:
    r = SESSION.get(USDM, params={
        "aoi": fips, "startdate": "1/1/2006", "enddate": "12/31/2024",
        "statisticsType": "1"}, timeout=120)
    if r.status_code != 200 or not r.text.strip():
        return None
    df = pd.read_csv(io.StringIO(r.text))
    if "MapDate" not in df.columns or not len(df):
        return None
    df["date"] = pd.to_datetime(df["MapDate"], format="%Y%m%d")
    time.sleep(0.3)
    return df


class DroughtSeries:
    """SPI percentiles for one POWER precip cell."""

    def __init__(self, raw: dict):
        t = raw["time"]
        self.idx = {d: i for i, d in enumerate(t)}
        self.doy = np.array([int(d[5:7]) * 31 + int(d[8:10]) for d in t])
        pr = np.array([0.0 if v is None else v for v in raw["precipitation_sum"]])
        self._cum = np.concatenate([[0.0], np.cumsum(pr)])
        self._spi = {}
        for w in WINDOWS:
            ws = self._cum[w:] - self._cum[:-w]
            ws = np.concatenate([np.full(w - 1, np.nan), ws])
            out = np.full(len(t), np.nan)
            fin = np.isfinite(ws)
            for tgt in np.unique(self.doy):
                m = fin & (np.abs(self.doy - tgt) <= 15)
                hs = np.sort(ws[m])
                if hs.size >= 30:
                    ii = np.where(self.doy == tgt)[0]
                    out[ii] = np.searchsorted(hs, ws[ii], side="right") / hs.size
            self._spi[w] = out

    def features(self, date: str) -> dict | None:
        i = self.idx.get(date)
        if i is None or i < max(WINDOWS):
            return None
        f = {f"spi{w}": float(self._spi[w][i]) for w in WINDOWS}
        j = max(0, i - 56)
        prev = self._spi[90][j]
        f["spi90_trend_8w"] = (float(self._spi[90][i] - prev)
                               if np.isfinite(prev) else 0.0)
        return f if all(np.isfinite(v) for v in f.values()) else None


def main():
    counties = pick_counties()
    rows = []
    per_county = []
    for c in counties.itertuples(index=False):
        u = usdm_series(c.GEOID)
        if u is None or len(u) < 400:
            continue
        raw = nasapower.fetch_cell(*nasapower.cell(c.lat, c.lon))
        if raw is None:
            continue
        ds = DroughtSeries(raw)
        cc = []
        for r in u.itertuples(index=False):
            f = ds.features(r.date.strftime("%Y-%m-%d"))
            if f is None:
                continue
            sev = (r.D1 + r.D2 + r.D3 + r.D4) / 4.0        # graded severity 0-100
            cc.append({"fips": c.GEOID, "state": c.USPS,
                       "date": r.date.strftime("%Y-%m-%d"),
                       "d2_area": float(r.D2), "severity": float(sev),
                       "label": int(r.D2 >= D2_THRESHOLD), **f})
        if len(cc) < 400:
            continue
        cd = pd.DataFrame(cc)
        rows.append(cd)
        if cd["label"].nunique() > 1:
            auc = roc_auc_score(cd["label"], 1 - cd["spi90"])
            rho = spearmanr(cd["severity"], -cd["spi90"]).statistic
            per_county.append({"fips": c.GEOID, "state": c.USPS,
                               "n_weeks": len(cd),
                               "d2_weeks": int(cd.label.sum()),
                               "auc_spi90": float(auc),
                               "spearman_sev_spi90": float(rho)})
            print(f"  {c.USPS} {c.GEOID}  weeks={len(cd):4d}  "
                  f"D2+={int(cd.label.sum()):4d}  AUC(1-spi90)={auc:.3f}  "
                  f"rho={rho:+.3f}", flush=True)
        else:
            per_county.append({"fips": c.GEOID, "state": c.USPS,
                               "n_weeks": len(cd),
                               "d2_weeks": int(cd.label.sum()),
                               "auc_spi90": None, "spearman_sev_spi90": None})

    d = pd.concat(rows, ignore_index=True)
    d.to_csv(PROCESSED / "drought_panel.csv", index=False)
    aucs = [p["auc_spi90"] for p in per_county if p["auc_spi90"]]
    rhos = [p["spearman_sev_spi90"] for p in per_county if p["spearman_sev_spi90"]]
    print(f"\npanel: {len(d):,} county-weeks, {d.label.mean():.3f} in D2+")
    print(f"AUC(1-spi90) across counties: median {np.median(aucs):.3f} "
          f"IQR [{np.percentile(aucs,25):.3f}, {np.percentile(aucs,75):.3f}]")
    print(f"Spearman(severity, -spi90):  median {np.median(rhos):+.3f}")

    # pooled drought head: SPI features -> P(county mostly in D2+),
    # grouped CV by county so no county grades itself
    X = d[FEATURES].to_numpy("float64")
    y = d["label"].to_numpy()
    g = d["fips"].to_numpy()
    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        m = LGBMClassifier(objective="binary", n_estimators=300,
                           learning_rate=0.05, num_leaves=15, verbose=-1)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    fin = np.isfinite(oof)
    head = {"roc_auc": float(roc_auc_score(y[fin], oof[fin])),
            "pr_auc": float(average_precision_score(y[fin], oof[fin])),
            "base_rate": float(y.mean())}
    print(f"drought head (county-grouped CV): ROC {head['roc_auc']:.3f}  "
          f"PR {head['pr_auc']:.3f} (base {head['base_rate']:.3f})")

    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[fin], y[fin])
    final = LGBMClassifier(objective="binary", n_estimators=300,
                           learning_rate=0.05, num_leaves=15, verbose=-1).fit(X, y)
    with (ROOT / "models" / "artifacts" / "drought_head.pkl").open("wb") as fh:
        pickle.dump({"model": final, "calibrator": iso, "features": FEATURES,
                     "target": f"county >=50% in USDM D2+",
                     "caveat": "calibrated on CONUS truth only"}, fh)

    (RUNS / "drought-validation.json").write_text(json.dumps({
        "name": "drought-validation", "layer": "drought",
        "indicator": "empirical SPI (30/90/180d seasonal percentile of precip)",
        "truth": "US Drought Monitor county weekly area-% (open API)",
        "n_counties": len(per_county), "n_county_weeks": int(len(d)),
        "auc_spi90_median": float(np.median(aucs)),
        "auc_spi90_iqr": [float(np.percentile(aucs, 25)),
                          float(np.percentile(aucs, 75))],
        "spearman_median": float(np.median(rhos)),
        "drought_head_cv": head,
        "per_county": per_county,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2, default=float), encoding="utf-8")
    print("wrote models/runs/drought-validation.json + artifacts/drought_head.pkl")


if __name__ == "__main__":
    main()
