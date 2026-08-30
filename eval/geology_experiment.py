"""Does substrate (soil texture + lithology) help the susceptibility layer?

Three questions, asked in the order that makes a negative answer cheap to
believe:

  (a) LOCAL   per-region spatial CV, terrain-only vs terrain+geology.
              If geology carries real signal it should show up here first.
  (b) TRANSFER pnw+brazil -> myanmar, pnw+myanmar -> brazil. D16 found regional
              terrain models do not travel (ROC 0.47 on Myanmar) because the
              learned weighting is regime-specific. Substrate is a candidate
              fix: "weathered clay-rich regolith on steep ground fails" ought
              to mean the same thing on two continents, where "elevation 1800m"
              does not.
  (c) OSO     the D24 miss. Deep-seated failure in glacial outwash, terrain-only
              susceptibility ~0.02-0.06 at a site that killed 43 people. Does
              knowing the ground is unconsolidated sediment move it?

Expectations written down BEFORE running, per the house rule (D24):
  (a) small positive, maybe +0.01-0.03 ROC. Texture is a weak proxy for the
      geotechnical properties that matter, at 250 m.
  (b) genuinely uncertain, and the interesting one. Could go either way.
  (c) probably NOT. SoilGrids models the top 30 cm of SOIL; Oso failed on a
      ~200 m deep-seated surface in glacial outwash beneath it. GLiM at 0.5 deg
      puts Oso in a cell labelled metamorphic (the Cascades crystalline core),
      not the valley fill. Both sources are thematically or spatially wrong for
      this failure mode. If the number does not move, that is the finding.

Everything is untuned BASE_PARAMS + GroupKFold(5) on block_id, so the
terrain-only and terrain+geology arms differ only in their feature columns.

Stages:  --stage annotate     write data/processed/region_{name}_geo.csv
         --stage experiments  run (a)(b)(c), write models/runs/geology-experiment.json
         --stage all          both
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, ROOT  # noqa: E402
from eval.spatial_cv import spatial_cv, summarise  # noqa: E402
from features import terrain  # noqa: E402
from models.train import BASE_PARAMS  # noqa: E402
from pipelines import geology  # noqa: E402

REGIONS = ["pnw", "myanmar", "brazil"]
TERRAIN = list(terrain.FEATURES)
GEO = list(geology.FEATURES)
CAT = "lith_class"
N_SPLITS = 5
RUNS = ROOT / "models" / "runs"
OSO = (48.2836, -121.8477)
# house convention for the "~900 m ring" (serve/score_global.py): a 3x3
# lattice at +/-0.008 deg. At 48N that is 890 m north-south, 596 m east-west.
RING_OFFSETS = [(dla, dlo) for dla in (-0.008, 0.0, 0.008)
                for dlo in (-0.008, 0.0, 0.008)]

SOIL = list(geology.SOIL_FEATURES)      # SoilGrids only, no lithology

# terrain+soil exists to isolate lithology's MARGINAL contribution: with GLiM
# only available at 0.5 deg, "terrain+geo vs terrain+soil" is the direct test
# of whether a 55 km lithology cell adds anything over 250 m soil texture.
ARMS = {
    "terrain": TERRAIN,
    "terrain+soil": TERRAIN + SOIL,
    "terrain+geo": TERRAIN + GEO,
    "geo_only": GEO,
}


def gain_breakdown(model, feats: list[str]) -> dict:
    """Gain share carried by soil texture vs lithography vs terrain."""
    imp = importance(model, feats)
    by = {d["feature"]: d["gain_pct"] for d in imp}
    return {
        "soil_gain_pct": float(sum(v for f, v in by.items() if f in SOIL)),
        "lith_gain_pct": float(by.get(CAT, 0.0)),
        "geo_gain_pct": float(sum(v for f, v in by.items() if f in GEO)),
        "terrain_gain_pct": float(sum(v for f, v in by.items() if f in TERRAIN)),
        "per_geo_feature": {f: float(by[f]) for f in GEO if f in by},
    }


def geo_csv(name: str) -> Path:
    return PROCESSED / f"region_{name}_geo.csv"


# ---------------------------------------------------------------- annotate --

def annotate(name: str) -> None:
    """Copy the region matrix and add geology columns. Originals untouched."""
    src = PROCESSED / f"region_{name}.csv"
    out = geo_csv(name)
    if out.exists():
        print(f"  [{name}] cached ({out.name})")
        return
    df = pd.read_csv(src)
    rows = [geology.geo_features(la, lo)
            for la, lo in zip(df["lat"].to_numpy(), df["lon"].to_numpy())]
    g = pd.DataFrame(rows, index=df.index)
    for c in GEO:
        df[c] = g[c]
    df.to_csv(out, index=False)
    geology.clear_cache()
    cov = {c: float(df[c].notna().mean()) for c in GEO}
    print(f"  [{name}] {len(df):,} rows -> {out.name}")
    print("           coverage " + "  ".join(f"{c}={cov[c]:.3f}" for c in GEO))


# ------------------------------------------------------------------ models --

def matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Feature matrix. LightGBM reads negatives in a categorical column as
    missing, so the lithology NaNs become -1 rather than being imputed."""
    X = df[feats].to_numpy("float64", copy=True)
    if CAT in feats:
        j = feats.index(CAT)
        X[~np.isfinite(X[:, j]), j] = -1
    return X


def cat_idx(feats: list[str]) -> list[int]:
    return [feats.index(CAT)] if CAT in feats else []


def make_fit_predict(feats: list[str], params: dict | None = None):
    p = dict(params or BASE_PARAMS)
    cats = cat_idx(feats)

    def fit_predict(Xtr, ytr, Xte):
        m = LGBMClassifier(**p)
        if cats:
            m.fit(Xtr, ytr, categorical_feature=cats)
        else:
            m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    return fit_predict


def fit_full(df: pd.DataFrame, feats: list[str]):
    m = LGBMClassifier(**BASE_PARAMS)
    X, y = matrix(df, feats), df["label"].to_numpy()
    cats = cat_idx(feats)
    if cats:
        m.fit(X, y, categorical_feature=cats)
    else:
        m.fit(X, y)
    return m


def importance(model, feats: list[str]) -> list[dict]:
    g = model.booster_.feature_importance(importance_type="gain")
    tot = float(g.sum()) or 1.0
    return sorted(({"feature": f, "gain_pct": 100.0 * float(v) / tot}
                   for f, v in zip(feats, g)), key=lambda d: -d["gain_pct"])


# ------------------------------------------------------------- (a) local CV --

def local_cv(dfs: dict) -> dict:
    print("\n(a) per-region local CV, GroupKFold(5) on block_id")
    res: dict = {}
    for name in REGIONS:
        df = dfs[name]
        res[name] = {"n_rows": int(len(df)), "n_pos": int(df["label"].sum()),
                     "base_rate": float(df["label"].mean()),
                     "n_blocks": int(df["block_id"].nunique()), "arms": {}}
        for arm, feats in ARMS.items():
            X = matrix(df, feats)
            y = df["label"].to_numpy()
            gsplit = df["block_id"].to_numpy()
            folds, oof = spatial_cv(X, y, gsplit, make_fit_predict(feats),
                                    n_splits=N_SPLITS)
            s = summarise(folds, oof, y)
            m = fit_full(df, feats)
            res[name]["arms"][arm] = {
                "features": feats,
                "roc_auc_mean": s["roc_auc_mean"],
                "pr_auc_mean": s["pr_auc_mean"],
                "pr_auc_sd": s["pr_auc_sd"],
                "lift_mean": s["lift_mean"],
                "pooled_oof_roc": s["pooled_oof"].get("roc_auc"),
                "pooled_oof_pr": s["pooled_oof"].get("pr_auc"),
                "importance_top": importance(m, feats)[:8],
                **gain_breakdown(m, feats),
            }
            print(f"  {name:9} {arm:12} ROC {s['roc_auc_mean']:.4f}  "
                  f"PR {s['pr_auc_mean']:.4f} (sd {s['pr_auc_sd']:.4f})  "
                  f"lift {s['lift_mean']:.2f}x")
        a = res[name]["arms"]["terrain"]
        b = res[name]["arms"]["terrain+geo"]
        c = res[name]["arms"]["terrain+soil"]
        res[name]["delta_roc"] = b["roc_auc_mean"] - a["roc_auc_mean"]
        res[name]["delta_pr"] = b["pr_auc_mean"] - a["pr_auc_mean"]
        res[name]["delta_roc_soil_only"] = c["roc_auc_mean"] - a["roc_auc_mean"]
        # what lithology adds ON TOP of soil texture -- the 0.5 deg question
        res[name]["delta_roc_lith_marginal"] = b["roc_auc_mean"] - c["roc_auc_mean"]
        print(f"  {name:9} {'DELTA':12} ROC {res[name]['delta_roc']:+.4f}  "
              f"PR {res[name]['delta_pr']:+.4f}   "
              f"(geology {b['geo_gain_pct']:.1f}% of gain: "
              f"soil {b['soil_gain_pct']:.1f}%, lith {b['lith_gain_pct']:.1f}%)")
        print(f"  {name:9} {'':12} lithology marginal over soil-only: "
              f"{res[name]['delta_roc_lith_marginal']:+.4f} ROC")
    return res


# -------------------------------------------------------------- (b) transfer --

def shift_diagnosis(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """Why a pooled model fails on a held-out region, in D16's terms.

    D16 diagnosed the terrain transfer failure as covariate shift: 57-63% of
    Myanmar lay beyond the PNW's 90th percentile. The same question is asked
    here of the substrate features, plus the one that only categoricals have --
    how much of the test region sits in a lithology class the model never saw.
    """
    out: dict = {}
    for f in SOIL:
        a = tr[f].dropna().to_numpy()
        b = te[f].dropna().to_numpy()
        if len(a) < 10 or len(b) < 10:
            continue
        lo, hi = np.percentile(a, 5), np.percentile(a, 95)
        out[f] = {
            "train_mean": float(a.mean()), "test_mean": float(b.mean()),
            "frac_test_outside_train_5_95": float(((b < lo) | (b > hi)).mean()),
        }
    seen = set(tr[CAT].dropna().astype(int).unique())
    lith = te[CAT].dropna().astype(int)
    unseen_mass = float((~lith.isin(seen)).mean()) if len(lith) else float("nan")
    unseen = sorted({geology.GLIM_CLASSES.get(int(v), str(v))
                     for v in lith.unique() if int(v) not in seen})
    out["lithology"] = {
        "train_classes": sorted(geology.GLIM_CLASSES.get(int(v), str(v)) for v in seen),
        "frac_test_rows_in_unseen_class": unseen_mass,
        "unseen_classes_in_test": unseen,
    }
    return out


def transfer(dfs: dict, local: dict) -> dict:
    print("\n(b) cross-region transfer, trained cold on two regions")
    plans = [(["pnw", "brazil"], "myanmar"), (["pnw", "myanmar"], "brazil")]
    res: dict = {}
    for train_on, test_on in plans:
        key = f"{'+'.join(train_on)}->{test_on}"
        te = dfs[test_on]
        y_te = te["label"].to_numpy()
        entry = {"train_regions": train_on, "test_region": test_on,
                 "n_train": int(sum(len(dfs[r]) for r in train_on)),
                 "n_test": int(len(te)), "base_rate": float(y_te.mean()),
                 "arms": {}}
        tr = pd.concat([dfs[r] for r in train_on], ignore_index=True)
        for arm, feats in ARMS.items():
            m = fit_full(tr, feats)
            p = m.predict_proba(matrix(te, feats))[:, 1]
            entry["arms"][arm] = {
                "roc_auc": float(roc_auc_score(y_te, p)),
                "pr_auc": float(average_precision_score(y_te, p)),
                **gain_breakdown(m, feats),
            }
            print(f"  {key:26} {arm:12} ROC {entry['arms'][arm]['roc_auc']:.4f}"
                  f"  PR {entry['arms'][arm]['pr_auc']:.4f}")
        # floor and ceiling for context, exactly as D19 frames them
        entry["slope_only_roc"] = float(roc_auc_score(y_te, te["slope_deg"].to_numpy()))
        entry["local_ceiling_roc"] = local[test_on]["arms"]["terrain+geo"]["roc_auc_mean"]
        entry["delta_roc"] = (entry["arms"]["terrain+geo"]["roc_auc"]
                              - entry["arms"]["terrain"]["roc_auc"])
        entry["delta_roc_lith_marginal"] = (entry["arms"]["terrain+geo"]["roc_auc"]
                                            - entry["arms"]["terrain+soil"]["roc_auc"])
        print(f"  {key:26} {'DELTA':12} ROC {entry['delta_roc']:+.4f}   "
              f"(slope floor {entry['slope_only_roc']:.4f}, "
              f"local ceiling {entry['local_ceiling_roc']:.4f})")
        print(f"  {key:26} {'':12} geology gain in trained model: "
              f"soil {entry['arms']['terrain+geo']['soil_gain_pct']:.1f}%, "
              f"lith {entry['arms']['terrain+geo']['lith_gain_pct']:.1f}%; "
              f"lith marginal {entry['delta_roc_lith_marginal']:+.4f}")
        entry["diagnosis"] = shift_diagnosis(tr, te)
        d = entry["diagnosis"]["lithology"]
        print(f"  {key:26} {'':12} {d['frac_test_rows_in_unseen_class']*100:.1f}% of "
              f"test rows sit in a lithology class absent from training "
              f"{d['unseen_classes_in_test']}")
        print(f"  {key:26} {'':12} clay outside train 5-95 pctl: "
              f"{entry['diagnosis']['clay_pct']['frac_test_outside_train_5_95']*100:.1f}%"
              f" (train mean {entry['diagnosis']['clay_pct']['train_mean']:.1f}%, "
              f"test {entry['diagnosis']['clay_pct']['test_mean']:.1f}%)")
        res[key] = entry
    return res


# ------------------------------------------------------------------ (c) Oso --

def calibrated_pnw(df: pd.DataFrame, feats: list[str]):
    """Model + isotonic calibrator fitted on out-of-fold predictions only.

    The D24 number (~0.02) is a calibrated probability, so an uncalibrated raw
    score here would not be comparable.
    """
    X, y = matrix(df, feats), df["label"].to_numpy()
    folds, oof = spatial_cv(X, y, df["block_id"].to_numpy(),
                            make_fit_predict(feats), n_splits=N_SPLITS)
    ok = np.isfinite(oof)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[ok], y[ok])
    return fit_full(df, feats), iso


def oso_probe(dfs: dict) -> dict:
    print("\n(c) Oso probe (48.2836, -121.8477), pnw-trained, isotonic-calibrated")
    df = dfs["pnw"]
    pts = [(OSO[0] + dla, OSO[1] + dlo) for dla, dlo in RING_OFFSETS]
    feat_rows, kept = [], []
    for la, lo in pts:
        t = terrain.derive(la, lo)
        if t is None:
            continue
        row = dict(t)
        row.update(geology.geo_features(la, lo))
        feat_rows.append(row)
        kept.append((la, lo))
    if not feat_rows:
        return {"error": "no DEM coverage at Oso"}
    probe = pd.DataFrame(feat_rows)
    centre_i = next(i for i, (la, lo) in enumerate(kept)
                    if abs(la - OSO[0]) < 1e-9 and abs(lo - OSO[1]) < 1e-9)

    res: dict = {"lat": OSO[0], "lon": OSO[1],
                 "ring": {"offsets_deg": 0.008, "n_points": len(kept),
                          "note": "3x3 lattice; 890 m N-S, 596 m E-W at 48N"},
                 "geology_at_site": {}, "arms": {}}
    gs = geology.geo_features(*OSO)
    lc = gs["lith_class"]
    code = geology.GLIM_CLASSES.get(int(lc)) if np.isfinite(lc) else None
    res["geology_at_site"] = {k: (None if not np.isfinite(v) else round(float(v), 2))
                              for k, v in gs.items()}
    res["geology_at_site"]["lith_code"] = code
    res["geology_at_site"]["lith_name"] = geology.GLIM_LONG.get(code)

    for arm, feats in ARMS.items():
        m, iso = calibrated_pnw(df, feats)
        raw = m.predict_proba(matrix(probe, feats))[:, 1]
        cal = iso.predict(raw)
        res["arms"][arm] = {
            "point": float(cal[centre_i]),
            "ring_max": float(np.max(cal)),
            "ring_mean": float(np.mean(cal)),
            "point_raw": float(raw[centre_i]),
            "ring_max_raw": float(np.max(raw)),
            # the deployed gate reads the NEIGHBOURHOOD max, not the point
            # (serve/score_global.py: susc_near >= 0.3), so this is the number
            # that would actually decide whether Oso alerts
            "clears_susceptibility_gate_0.30": bool(np.max(cal) >= 0.30),
        }
        print(f"  {arm:12} point {cal[centre_i]:.4f}   ring-max {np.max(cal):.4f}"
              f"   (raw {raw[centre_i]:.4f} / {np.max(raw):.4f})")
    res["gate"] = {
        "threshold": 0.30, "source": "serve/score_global.py susc_near >= 0.3",
        "note": ("the product gates on the ~900 m neighbourhood max, so ring_max "
                 "is the operationally decisive number, not the point score"),
    }
    res["d24_baseline_note"] = (
        "D24 records susceptibility 0.023 at Oso, but that is the DEPLOYED "
        "product model (different training matrix, road-aware features, tuned "
        "params, its own calibration). The terrain-only arm here is trained on "
        "region_pnw.csv with untuned BASE_PARAMS at a 16.7% base rate and is "
        "NOT comparable to 0.023. Only the within-experiment terrain vs "
        "terrain+geo contrast is a valid comparison.")
    res["delta_point"] = (res["arms"]["terrain+geo"]["point"]
                          - res["arms"]["terrain"]["point"])
    res["delta_ring_max"] = (res["arms"]["terrain+geo"]["ring_max"]
                             - res["arms"]["terrain"]["ring_max"])
    print(f"  {'DELTA':12} point {res['delta_point']:+.4f}   "
          f"ring-max {res['delta_ring_max']:+.4f}")
    print(f"  substrate at Oso: clay {res['geology_at_site']['clay_pct']}%  "
          f"sand {res['geology_at_site']['sand_pct']}%  "
          f"lith {code} ({geology.GLIM_LONG.get(code)})")
    return res


# -------------------------------------------------------------------- table --

def print_table(local: dict, tr: dict, oso: dict) -> None:
    print("\n" + "=" * 78)
    print("GEOLOGY / SOIL FEATURES -- RESULTS")
    print("=" * 78)
    print("\n(a) local CV, ROC-AUC / PR-AUC, GroupKFold(5) on block_id")
    print(f"  {'region':<9} {'terrain':>13} {'terr+soil':>13} {'terr+geo':>13} "
          f"{'geo only':>13} {'dROC':>7}")
    for r in REGIONS:
        a = local[r]["arms"]
        cells = "".join(
            f"{a[k]['roc_auc_mean']:>6.3f}/{a[k]['pr_auc_mean']:<6.3f}"
            for k in ("terrain", "terrain+soil", "terrain+geo", "geo_only"))
        print(f"  {r:<9} {cells} {local[r]['delta_roc']:>+7.3f}")
    print("\n  gain share in the terrain+geo model, and what lithology adds"
          " over soil alone:")
    print(f"  {'region':<9} {'terrain%':>9} {'soil%':>8} {'lith%':>8} "
          f"{'dROC soil':>10} {'dROC lith':>10}")
    for r in REGIONS:
        b = local[r]["arms"]["terrain+geo"]
        print(f"  {r:<9} {b['terrain_gain_pct']:>9.1f} {b['soil_gain_pct']:>8.1f} "
              f"{b['lith_gain_pct']:>8.1f} "
              f"{local[r]['delta_roc_soil_only']:>+10.3f} "
              f"{local[r]['delta_roc_lith_marginal']:>+10.3f}")

    print("\n(b) transfer, ROC-AUC on the held-out region (trained cold)")
    print(f"  {'plan':<24} {'terrain':>8} {'t+soil':>8} {'t+geo':>8} {'d':>7} "
          f"{'slope':>7} {'local':>7}")
    for k, e in tr.items():
        print(f"  {k:<24} {e['arms']['terrain']['roc_auc']:>8.3f} "
              f"{e['arms']['terrain+soil']['roc_auc']:>8.3f} "
              f"{e['arms']['terrain+geo']['roc_auc']:>8.3f} "
              f"{e['delta_roc']:>+7.3f} {e['slope_only_roc']:>7.3f} "
              f"{e['local_ceiling_roc']:>7.3f}")
    print("\n(c) Oso 2014 (48.2836, -121.8477), pnw model, calibrated")
    if "error" in oso:
        print("  " + oso["error"])
        return
    print(f"  {'arm':<14} {'point':>8} {'ring-max':>10} {'gate>=0.30':>12}")
    for arm in ARMS:
        a = oso["arms"][arm]
        print(f"  {arm:<14} {a['point']:>8.4f} {a['ring_max']:>10.4f} "
              f"{('CLEARS' if a['clears_susceptibility_gate_0.30'] else 'fails'):>12}")
    print(f"  {'DELTA':<14} {oso['delta_point']:>+8.4f} "
          f"{oso['delta_ring_max']:>+10.4f}")
    g = oso["geology_at_site"]
    print(f"  substrate reported at Oso: clay {g['clay_pct']}%, silt {g['silt_pct']}%, "
          f"sand {g['sand_pct']}%, lithology {g['lith_code']} ({g['lith_name']})")
    print("  ring-max is what the product gates on (serve/score_global.py).")


# --------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["annotate", "experiments", "all"],
                    default="all")
    a = ap.parse_args()

    if a.stage in ("annotate", "all"):
        print("annotating region matrices with geology columns")
        for r in REGIONS:
            annotate(r)
    if a.stage == "annotate":
        return

    dfs = {r: pd.read_csv(geo_csv(r)) for r in REGIONS}
    cov = {}
    for r in REGIONS:
        d = dfs[r]
        lith = d[CAT].dropna().astype(int)
        cov[r] = {
            "coverage": {c: float(d[c].notna().mean()) for c in GEO},
            "lith_classes_present": {geology.GLIM_CLASSES.get(int(k), str(k)): int(v)
                                     for k, v in lith.value_counts().items()},
            "clay_pct_mean": float(d["clay_pct"].mean()),
            "sand_pct_mean": float(d["sand_pct"].mean()),
            "missingness_confound": missingness_confound(d),
        }

    local = local_cv(dfs)
    tr = transfer(dfs, local)
    oso = oso_probe(dfs)
    print_table(local, tr, oso)

    run = {
        "name": "geology-experiment",
        "layer": "susceptibility",
        "model": "LightGBM (BASE_PARAMS, untuned)",
        "status": "complete",
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "question": ("do soil-texture and lithology features improve regional "
                     "susceptibility models, cross-region transfer (D16), and "
                     "the Oso deep-seated miss (D24)"),
        "regions": REGIONS,
        "cv": {"scheme": "GroupKFold on block_id", "n_splits": N_SPLITS},
        "params": {k: v for k, v in BASE_PARAMS.items()},
        "feature_sets": ARMS,
        "region_geology_summary": cov,
        "data_sources": DATA_SOURCES,
        "results_local_cv": local,
        "results_transfer": tr,
        "results_oso": oso,
        "caveats": CAVEATS,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "geology-experiment.json"
    out.write_text(json.dumps(run, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"geology cache: {geology.cache_bytes()/1e6:.1f} MB")


DATA_SOURCES = [
    {
        "name": "SoilGrids 250m v2.0",
        "provider": "ISRIC - World Soil Information",
        "license": "CC-BY 4.0",
        "license_verified": ("isric.org/explore/soilgrids and ISRIC Data and "
                             "Software Policy; WCS GetCapabilities reports "
                             "AccessConstraints=None, Fees=None"),
        "citation": ("Poggio, L., de Sousa, L.M., Batjes, N.H., Heuvelink, "
                     "G.B.M., Kempen, B., Ribeiro, E., Rossiter, D. (2021). "
                     "SoilGrids 2.0: producing soil information for the globe "
                     "with quantified spatial uncertainty. SOIL 7, 217-240."),
        "access": ("WCS 2.0.1 GetCoverage windows from maps.isric.org, "
                   "SUBSETTINGCRS/OUTPUTCRS EPSG:4326, no key, no account"),
        "layers": ["clay 0-30cm", "sand 0-30cm", "bdod 0-30cm", "cfvo 0-30cm",
                   "silt derived as 1000 - clay - sand (g/kg)"],
        "depth_handling": "thickness-weighted mean of 0-5, 5-15, 15-30 cm",
        "native_resolution_m": 250,
    },
    {
        "name": "GLiM Global Lithological Map v1.0 (0.5 degree gridded)",
        "provider": "Hartmann, J. & Moosdorf, N., University of Hamburg / PANGAEA",
        "license": "CC-BY 3.0",
        "license_verified": "doi.pangaea.de/10.1594/PANGAEA.788537 states CC-BY-3.0",
        "citation": ("Hartmann, J., Moosdorf, N. (2012). The new global "
                     "lithological map database GLiM: a representation of rock "
                     "properties at the Earth surface. Geochem. Geophys. "
                     "Geosyst. 13, Q12004. doi:10.1029/2012GC004370"),
        "access": "hdl.handle.net/10013/epic.39939.d001, 38 kB zip, no key",
        "native_resolution": ("0.5 degree (~55 km). The full 1,235,400-polygon "
                             "shapefile is distributed via CCGM.ORG, a "
                             "commercial publisher, and is NOT reachable under "
                             "the project's public/keyless constraint."),
    },
]

CAVEATS = [
    "GLiM lith_class is 0.5 degree (~55 km cells). Myanmar's 2x2 deg box holds "
    "~16 cells; within-region variation is minimal by construction, so a null "
    "result for lithology here is a resolution result, not a geology result.",
    "SoilGrids models the top 30 cm of SOIL. Deep-seated landslides fail metres "
    "to hundreds of metres below that. The Oso probe is therefore a test of a "
    "known-imperfect proxy, and was expected in advance to fail.",
    "Untuned BASE_PARAMS throughout so the arms differ only in feature columns; "
    "absolute scores are below the tuned numbers reported elsewhere.",
    "PNW labels carry the D10/D11 reporting bias; its terrain-only baseline here "
    "is the portable terrain set, not the road-aware product model.",
    "Texture fractions are compositional (clay+sand+silt=100), so the three are "
    "collinear by construction; gain shares split arbitrarily between them.",
    "PNW SoilGrids coverage is only 80.3% (coastal/water cells are unmapped), and "
    "missingness is NOT random with respect to the label: 23.5% of positives vs "
    "18.9% of background, with missing cells averaging 80 m elevation against "
    "377 m. Missingness is therefore a weak proxy for the low-elevation coastal "
    "reporting artifact of D10/D11 (ROC 0.523 on its own), so some unknown part "
    "of the PNW gain is that artifact rather than substrate physics. Myanmar "
    "(100% coverage) and Brazil (97.9%) are clean; Myanmar is the trustworthy "
    "positive result.",
    "Single-point scores are noisy from a ~0.8-ROC model -- D11's warning about "
    "judging a probabilistic model by hand-picked points applies to the Oso "
    "numbers, which is why the ring and the population results are reported "
    "alongside them.",
]


def missingness_confound(df: pd.DataFrame) -> dict:
    """Is 'SoilGrids has no data here' itself predicting the label? (D10/D11)"""
    m = df["clay_pct"].isna()
    pos, bg = df["label"] == 1, df["label"] == 0
    out = {"missing_rate": float(m.mean()),
           "missing_rate_positives": float(m[pos].mean()),
           "missing_rate_background": float(m[bg].mean())}
    if 0 < m.mean() < 1:
        out["roc_of_missingness_alone"] = float(
            roc_auc_score(df["label"].to_numpy(), m.to_numpy().astype(int)))
        out["mean_elev_missing"] = float(df.loc[m, "elev_m"].mean())
        out["mean_elev_present"] = float(df.loc[~m, "elev_m"].mean())
    return out

if __name__ == "__main__":
    main()
