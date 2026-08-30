"""Deployment-distribution thresholds for the fire danger layer.

D14's lesson applied to fire before it can bite: the fire model was trained at
a 1-case-per-5-rows base rate, so its scores mean nothing as alarms until
they are priced on the real distribution of days. This scores the 541 cached
North-American fire-weather cells on a weekly-sampled grid (every 7th day,
2006-2024 -- quantiles don't need every day) and records the score quantiles
for 2/5/10% alarm budgets, written to serve/thresholds.json alongside the
landslide trigger's.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT  # noqa: E402
from pipelines import fireweather as fw  # noqa: E402

BUDGETS = (0.02, 0.05, 0.10)
RECOMMENDED = 0.05




def tie_safe_threshold(scores, budget):
    """Threshold whose ACHIEVED alarm rate is <= budget under `>= thr` serving.

    Isotonic calibration emits step functions, so naive quantiles land exactly
    on plateau values shared by percent-scale masses of days; `>= thr` then
    admits the whole tied block and overspends the budget (D27: the 5% fire
    budget spent 8.3-14%). Pick the smallest DISTINCT score whose >=-rate fits
    the budget, and report the rate actually achieved.
    """
    import numpy as np
    s = np.sort(np.asarray(scores))
    n = len(s)
    uniq = np.unique(s)
    # rate(v) = P(score >= v) = (n - searchsorted_left(v)) / n
    rates = (n - np.searchsorted(s, uniq, side="left")) / n
    ok = rates <= budget
    if not ok.any():
        return float(uniq[-1]), float(rates[-1])
    i = int(np.argmax(ok))                    # smallest value fitting budget
    return float(uniq[i]), float(rates[i])

def main():
    with (ROOT / "models" / "artifacts" / "fire_trigger.pkl").open("rb") as fh:
        b = pickle.load(fh)
    model, iso, feats = b["model"], b["calibrator"], b["features"]

    cells = []
    for p in sorted(fw.FW_DIR.glob("*.json.gz")):
        la, lo = p.stem.replace(".json", "").split("_")
        cells.append((float(la), float(lo)))
    print(f"cells cached: {len(cells)}")

    scores = []
    for n, c in enumerate(cells, 1):
        s = fw.series_for(*c)
        if s is None:
            continue
        dates = [d for i, d in enumerate(s.idx) if i >= 730 and i % 7 == 0
                 and d >= "2006-01-01"]
        X = []
        for d in dates:
            f = s.features(d)
            if f:
                X.append([f.get(k, np.nan) for k in feats])
        if X:
            p = iso.predict(model.predict_proba(np.asarray(X))[:, 1])
            scores.append(p)
        if n % 50 == 0:
            print(f"  {n}/{len(cells)} cells", flush=True)
        fw._CACHE.clear()          # keep memory flat across 541 cells
    all_p = np.concatenate(scores)
    print(f"scored {len(all_p):,} weekly cell-samples")

    tj = ROOT / "serve" / "thresholds.json"
    cur = json.loads(tj.read_text(encoding="utf-8")) if tj.exists() else {}
    fire = {"computed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "samples": int(len(all_p)), "budgets": {}}
    for bud in BUDGETS:
        thr, achieved = tie_safe_threshold(all_p, bud)
        fire["budgets"][f"{bud:.2f}"] = {
            "threshold": thr,
            "achieved_alarm_rate": round(achieved, 5),
            "achieved_alarm_days_per_cell_year": round(achieved * 365.25, 1)}
        print(f"  budget {bud:.0%}: tie-safe threshold {thr:.4f} "
              f"achieves {achieved:.2%}")
    fire["threshold"] = fire["budgets"][f"{RECOMMENDED:.2f}"]["threshold"]
    cur["fire"] = fire
    tj.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    print(f"wrote fire thresholds -> {tj.relative_to(ROOT)} "
          f"(recommended {fire['threshold']:.4f})")


if __name__ == "__main__":
    main()
