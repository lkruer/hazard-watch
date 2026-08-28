# Multi-Hazard Risk Platform

Free, public, location-based hazard risk scoring. v1 builds the **landslide**
hazard end to end — ingest → features → model → spatial CV → serve — before any
other hazard is attempted, per `docs/CLAUDE.md`.

See `docs/decisions.md` for why each data source and modelling choice was made,
including the ones that differ from the brief and why.

## Layout

```
config.py            paths + committed study region
pipelines/           one file per data source, not per hazard
  common.py            retrying HTTP session + on-disk cache
  coolr.py             NASA COOLR landslide catalogues (ArcGIS FeatureServer)
  dem.py               Copernicus DEM GLO-30 tiles (public AWS S3, no auth)
  openmeteo.py         ERA5 daily rainfall + climatology percentiles
  select_region.py     picks the study region from label density
features/
  terrain.py           slope, aspect, curvature, TRI, roughness, relief, TPI
  sampling.py          background + case-crossover negatives, spatial blocks
  build_dataset.py     assembles both training matrices
eval/
  spatial_cv.py        GroupKFold on spatial blocks, PR-AUC, calibration
models/
  train.py             LightGBM, hyperparameter search, isotonic calibration
  runs/                one JSON per training run (feeds the dashboard)
  artifacts/           pickled model + calibrator
serve/
  score.py             combines both layers, writes flat JSON per location
reports/
  build_dashboard.py   renders reports/dashboard.html from on-disk state
docs/
  decisions.md         architecture decision log
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

## Run the pipeline

```bash
python pipelines/coolr.py                              # both catalogues
python pipelines/select_region.py                      # rank candidate regions
python features/build_dataset.py --stage all           # both matrices
python models/train.py --layer all --tune 40           # train + tune
python serve/score.py                                  # score demo locations
python reports/build_dashboard.py                      # refresh dashboard
```

Every network step caches to `data/cache/`, so re-runs are cheap and an
interrupted fetch resumes where it stopped.

## The two-layer design

| Layer | Question | Features | Negatives |
|---|---|---|---|
| **Susceptibility** | Where can this happen at all? | terrain only | background points across the region |
| **Trigger** | Is today unusual *for this place*? | rainfall anomaly only | same location, season-matched other dates |

Feature sets are kept **disjoint on purpose**. Giving the trigger model terrain
would let it relearn susceptibility, and the combined score would double-count
it. v1 combines them with the LHASA-style rule (both must cross their own
threshold) rather than a joint model — easier to debug, and it keeps "this
slope is vulnerable" and "today is unusual here" legible as separate claims.

## Reading the numbers

**PR-AUC is the headline, never accuracy.** These are rare-event problems where
"always predict no event" scores >83% accuracy on our sampled data and far
higher on a real daily grid. PR-AUC is always reported next to the base rate it
must beat, because PR-AUC alone is not comparable across datasets with
different positive fractions — the **lift over base rate** is the number that
means something.

**Cross-validation holds out whole 0.25° spatial blocks**, never random rows.
Two landslide points 300 m apart have near-identical features and outcomes; a
random split hands the model the answer and the score looks far better than it
is. `GroupKFold` on a block id, never `KFold`.

**Outputs are calibrated probabilities**, isotonic-fitted on out-of-fold
predictions only, and reported with expected calibration error — because the
product serves a probability, not a ranking.

## Known limits

- Susceptibility labels come from a report-derived catalogue, so events are
  over-reported near roads and towns. The susceptibility score is **relative
  within the study region**, not an absolute probability. The trigger layer is
  immune to this by construction (its controls are the same location).
- AlphaEarth embeddings are **not yet included** — they need Earth Engine
  authentication. The pipeline has a slot for them as one more susceptibility
  feature.
- ERA5 underestimates short convective rainfall extremes; mitigated by using
  percentiles against each cell's own record rather than absolute thresholds.

## Validation status (2026-08-28)

- **Prospective hindcast** (`eval/hindcast.py`): trigger frozen on ≤2015,
  replayed 2016–2024. At a 5%-of-days alarm budget it catches ~45% of reported
  events (±1 day). Thresholds are now set on the deployment distribution
  (`serve/calibrate_threshold.py`) after the training-set threshold proved to
  alarm on 39% of all days. A one-feature rain-percentile rule captures most
  of the model's skill — kept as the documented fallback.
- **External check** (`eval/external_check.py`): monotone agreement with
  NASA's global susceptibility map; ours is locally sharper on our labels.
- **Transfer test** (`eval/transfer.py`): the PNW terrain model scores
  *below chance* on Myanmar's bias-free satellite inventory, and four rescue
  attempts failed — regional susceptibility models do not travel. See
  `docs/decisions.md` D16.

## Target architecture: global, tiered by confidence

Global scope is reached by tiering, not by one world-model (D18):
**Tier A** regionally trained + hindcast-verified (PNW now);
**Tier B** global heuristics — NASA susceptibility map + the label-free
rain-percentile trigger (computable anywhere NASA POWER covers);
**Tier C** degraded inputs — still rendered, explicitly badged low-confidence,
never issuing red on data that cannot support it. Presentation: 2D map
(deferred; no UI yet).

## Not in scope, deliberately

No earthquake or tsunami prediction — no precursor signal exists for either,
and a trained "prediction" would produce a confident-looking number with no
skill. Those hazards appear, if at all, as exposure maps and relayed official
warnings only.
