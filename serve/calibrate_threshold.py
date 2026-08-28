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
        thr = float(np.quantile(all_p, 1 - bud))
        out["budgets"][f"{bud:.2f}"] = {
            "threshold": thr,
            "expected_alarm_days_per_cell_year": round(bud * 365.25, 1),
        }
        print(f"  budget {bud:.0%}: threshold {thr:.4f} "
              f"(~{bud*365.25:.0f} alarm days/cell/yr)")
    out["trigger_threshold"] = out["budgets"][f"{RECOMMENDED:.2f}"]["threshold"]

    p = ROOT / "serve" / "thresholds.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p.relative_to(ROOT)}  "
          f"(recommended trigger threshold {out['trigger_threshold']:.4f})")


if __name__ == "__main__":
    main()
