"""Deployment-distribution threshold calibration for the trigger layer.

The hindcast exposed a base-rate trap: an operating threshold tuned on the
case-control training set (events enriched ~500x over reality) alarmed on 39%
of all real days. A threshold only means something on the distribution it will
actually score -- so this script replays the FULL daily grid (every day
2004-2024 at every weather cell) through the production trigger model and
records the score quantiles that correspond to chosen alarm budgets.

"Alarm on the top 5% of days" is a statement a person can sanity-check
(~18 alarm days per year per cell, concentrated in the wet season); per the
2016-2024 prospective hindcast it catches ~45% of reported events. The 10%
budget (~37 days/yr) catches ~54%.

Writes serve/thresholds.json, which serve/score.py prefers over the
training-set operating point.
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
from eval.hindcast import features_all  # noqa: E402
from features.build_dataset import ACC_WEATHER, load_reports, region  # noqa: E402
from pipelines import nasapower  # noqa: E402
from pipelines.openmeteo import FEATURES, CellSeries  # noqa: E402

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
    with (ROOT / "models" / "artifacts" / "trigger.pkl").open("rb") as fh:
        b = pickle.load(fh)
    model, iso, feats = b["model"], b["calibrator"], b["features"]

    reg = region()
    ev = load_reports(reg["bbox"], ACC_WEATHER, need_date=True)
    ev = [e for e in ev if e["date"] >= nasapower.START]
    cells = sorted({nasapower.cell(e["lat"], e["lon"]) for e in ev})

    scores = []
    for c in cells:
        raw = nasapower.fetch_cell(*c)
        if raw is None:
            continue
        dates, X = features_all(CellSeries(raw))
        Xs = X[:, [FEATURES.index(f) for f in feats]]
        good = ~np.isnan(Xs).all(axis=1)
        p = iso.predict(model.predict_proba(Xs)[:, 1])
        scores.append(p[good])
    all_p = np.concatenate(scores)
    print(f"scored {len(all_p):,} cell-days across {len(cells)} cells")

    out = {
        "computed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "grid_cell_days": int(len(all_p)),
        "note": ("score quantiles of the production trigger model over the full "
                 "2004-2024 daily grid; 'alarm budget' 0.05 = alarm on the wettest "
                 "5% of days at these cells"),
        "budgets": {},
        "recommended_budget": RECOMMENDED,
    }
    for bud in BUDGETS:
        thr, achieved = tie_safe_threshold(all_p, bud)
        out["budgets"][f"{bud:.2f}"] = {
            "threshold": thr,
            "achieved_alarm_rate": round(achieved, 5),
            "achieved_alarm_days_per_cell_year": round(achieved * 365.25, 1),
        }
        print(f"  budget {bud:.0%}: tie-safe threshold {thr:.4f} "
              f"achieves {achieved:.2%} ({achieved*365.25:.0f} days/cell/yr)")
    out["trigger_threshold"] = out["budgets"][f"{RECOMMENDED:.2f}"]["threshold"]

    p = ROOT / "serve" / "thresholds.json"
    # merge, don't clobber -- other hazards' blocks (e.g. "fire") live here too
    cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    cur.update(out)
    out = cur
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p.relative_to(ROOT)}  "
          f"(recommended trigger threshold {out['trigger_threshold']:.4f})")


if __name__ == "__main__":
    main()
