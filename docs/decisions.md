# Architecture decisions — landslide (v1)

Running log of choices made while building M0–M4, with the evidence behind
them. The brief asks that data catalogue IDs be verified rather than assumed;
where a documented source had moved or died, that is recorded here.

---

## D1 — COOLR access path (the documented one is dead)

The brief points at COOLR "via data.nasa.gov / Landslide Viewer API". Verified
2026-08-27:

| Endpoint | Result |
|---|---|
| `data.nasa.gov/api/views/dd9e-wu2v/rows.csv` (legacy Socrata GLC export) | **404** |
| `data.nasa.gov/resource/dd9e-wu2v.json` | **404** |
| `maps.nccs.nasa.gov/arcgis/...` | connection reset |
| `gis.earthdata.nasa.gov/gis05/rest/services/Landslides/...` | **499 token required** |
| `gis.earthdata.nasa.gov/gis01/rest/services/Landslides/...` | **200 — anonymous** |

**Decision:** ingest from the `gis01` ArcGIS FeatureServer. Note the host number
matters — `gis05` is token-gated, `gis01` is not. Paginate at 2,000 rows
(`maxRecordCount`).

Also live on that server and worth using later: `Global_Landslide_Susceptibility`
(Stanley & Kirschbaum's heuristic global map) and `LHASA_Hazard_Today` /
`_Tomorrow` / `_Yesterday` — the operational NASA outputs, i.e. a free
benchmark to score our model against rather than guessing whether it is good.

## D2 — The two COOLR layers are not interchangeable

This is the finding that shaped the whole design.

| Layer | Rows | Structure |
|---|---|---|
| `COOLR_Events_Points` | 40,310 | Satellite-mapped **inventories**. Dense in space, almost degenerate in time. |
| `COOLR_Reports_Points` | 14,753 | Media/report **catalogue** (the classic GLC). Sparse in space, 3,664 distinct dates, 157 countries, 1974–2025. |

The inventories are clustered onto a handful of storm dates:

- Vietnam — 12,566 events on **11** distinct dates
- Myanmar — 7,972 events on **4**
- Mexico (SALaD-CD) — 3,862 events on **1**

**Decision:** map them onto the brief's two layers by their actual structure.
Inventories can support a susceptibility layer (a spatial question, where dates
do not matter) but cannot train a trigger layer — 11 dates is 11 independent
rainfall situations, and every point sharing a date shares its weather exactly.
The reports catalogue, with 3,664 dates, is the only trigger-capable source.

## D3 — Study region: Pacific Northwest (OR/WA)

Chosen empirically (`pipelines/select_region.py`) rather than by reputation, by
sliding a 2° window over both catalogues and scoring density and date-spread
separately.

| Region | Inventory events | Inventory dates | Reports | Report dates |
|---|---|---|---|---|
| Myanmar (Chin) | 7,970 | 2 | 106 | 40 |
| N Vietnam | 6,175 | 4 | 36 | 28 |
| Oregon Coast Range | 0 | 0 | 592 | **337** |
| Puget Sound | 0 | 0 | 321 | 196 |

No region carries both. The densest label sets cannot train a trigger; the best
trigger regions have no inventories.

**Decision:** `bbox = (-124.8, 42.0, -120.5, 49.2)` — 1,446 reports across
**635 distinct dates**, 831 of them located to "exact" accuracy. It is the only
box that supports both layers end to end, which the brief wants proven on one
hazard before generalising. The dense Myanmar/Vietnam inventories are kept as a
follow-on susceptibility enrichment, not discarded.

## D4 — Terrain: Copernicus DEM GLO-30 over AWS, not Earth Engine

The brief allows SRTM or Copernicus GLO-30. GLO-30 is served as public
Cloud-Optimized GeoTIFF from `copernicus-dem-30m.s3.amazonaws.com` with **no
credentials**, so the terrain layer is not blocked behind Earth Engine
onboarding. Tiles are 1°×1°, ~46 MB, named by south-west corner.

Metric-spacing note: a 1-arcsec pixel is ~30.9 m north–south but only
30.9·cos(lat) m east–west — ~21.7 m at 45°N. Differencing on raw degree spacing
inflates east–west gradients by 1/cos(lat) and biases aspect. `features/terrain.py`
converts to metres first.

## D5 — Rainfall: Open-Meteo ERA5 archive, not CHIRPS

The brief names CHIRPS or GPM IMERG. Both are reachable but awkward: CHIRPS
ships one global GeoTIFF **per day** (thousands of files for a multi-decade
record) and IMERG sits behind Earthdata Login. Open-Meteo serves ERA5/ERA5-Land
as point time series over plain HTTP, free and keyless — the same reanalysis
family the brief already nominates for the fire hazard.

**Tradeoff, recorded rather than hidden:** ERA5 underestimates short convective
extremes relative to gauge-blended CHIRPS, so absolute mm totals run low in
convective regimes. Mitigated by never using an absolute threshold — every
trigger feature is a percentile against that cell's own record, which is what
LHASA does and what the brief requires. Requests are deduplicated onto the
0.1° ERA5-Land grid, so hundreds of nearby labels collapse to one fetch.

Validation that the anomaly framing works, Oregon Coast Range:

| | 2015-12-08 (storm) | 2015-07-15 (dry) |
|---|---|---|
| 3-day precip | 136.1 mm | 0.5 mm |
| 3-day percentile (all year) | 0.998 | 0.326 |
| 3-day percentile (seasonal) | 0.988 | **0.720** |

The July trace is nothing in absolute terms but sits at the 72nd percentile
*for July* — precisely the distinction an absolute threshold destroys.

## D6 — Negative sampling: two designs, one per layer

Presence-only catalogue, so negatives are constructed. This is the easiest
place to manufacture a great-looking, worthless PR-AUC.

**Susceptibility — background sampling.** Points drawn across the region,
buffered 500 m from known events, latitude sampled equal-area. Known bias,
stated not hidden: COOLR is report-derived, so events are over-reported near
roads and towns, and the model partly learns *where people notice landslides*.
Treat the output as relative within-region, not an absolute probability.

**Trigger — case-crossover.** Controls are the **same location on other
dates**, season-matched (±45 days, other years), excluding ±7 days around any
real event there. Terrain, geology, land cover, road access and reporting
intensity are identical within a stratum, so none of them can separate case
from control — only weather varies. This removes the spatial reporting bias
from the trigger layer entirely. Season-matching stops the model cheating with
"it is winter", which is real but useless.

## D7 — The trigger model is not given terrain

Within a case-crossover stratum terrain is constant, so it cannot discriminate.
But *pooled* across strata, a model handed terrain would learn "steep places
have more events" — the susceptibility signal leaking into the trigger layer,
which would make the combined score double-count it. Feature sets are kept
disjoint: terrain → susceptibility, weather → trigger.

## D8 — Environment: Smart App Control

The machine had Windows Smart App Control enforcing
(`VerifiedAndReputablePolicyState = 1`). It refuses to load unsigned compiled
extensions, i.e. every binary wheel on PyPI: `pandas`, `scikit-learn`,
`lightgbm`, `pyarrow` and `numpy.random` all failed with *"An Application
Control policy has blocked this file"* while installing successfully.

Ingestion was unaffected — `requests` and the stdlib are pure Python — so M0
and M1 completed under the block. Resolved by the machine owner turning Smart
App Control off (now state `0`). No code changes were needed.

## D9 — Rate limits forced the trigger layer's shape

Open-Meteo's free tier prices a request by *locations × variables × days*. The
first attempt — 417 cells × 3 variables × 30 years — exhausted the hourly quota
outright (`"Hourly API request limit exceeded"`). Three changes, in order of
how much they bought:

1. **Multi-location batching.** The archive endpoint accepts comma-separated
   coordinates and returns one object per location *in request order*. Results
   must be matched by position, not by the coordinates echoed back, since the
   API snaps each request to the nearest ERA5 node. 20 cells per request
   instead of one: ~20× less wall-clock for the same data.
2. **Precipitation only.** Dropping `temperature_2m_max/min` cut the cost 3×.
   Rainfall is the dominant landslide trigger; temperature as a snowmelt proxy
   is a real but secondary driver and is deferred rather than paid for at
   triple the quota. `DAILY` plus a refetch restores it.
3. **Cells with ≥2 events only.** 218 of 417 cells, retaining 1,077 of 1,294
   events — **85% of the labels for 47% of the fetch**. A cell holding a single
   event contributes a single stratum, so the label lost per cell dropped is
   small.

Climatology was also shortened from 30 to 21 years (2004–2024): the region has
just 24 labelled events before 2005, so the extra decade cost a third of the
quota to buy 2% more labels.

The cache records the span it was fetched for, so widening `START`/`END` or
adding a variable invalidates stale entries instead of silently serving the
narrower series. Fetching sleeps to the top of the next hour on a quota error
and resumes from cache, so the pull is restartable.

## D10 — Reporting bias is real, measured, and ~30% of the headline score

The susceptibility model looked strong: PR-AUC **0.706** against a 0.167 base
rate (4.2× lift), ROC-AUC 0.887, under proper spatial block CV. Then gain-based
feature importance put `elev_m` at 31% (57% after tuning), which is not what
landslide physics predicts. Checking the distributions:

| elevation band | positives | background | enrichment |
|---|---|---|---|
| 0–100 m | **57.5%** | 11.3% | **5.1×** |
| 100–400 m | 30.0% | 23.9% | 1.3× |
| 400–1600 m | 11.6% | 54.9% | 0.21× |
| >1600 m | 0.3% | 8.5% | 0.04× |

Positives average 182 m against 756 m for background (standardised difference
−1.27), and TPI is negative for positives (−21.7 vs 0.5) — they sit in valley
bottoms. Steep, failure-prone ground is *more* common at altitude, not less.
This is the reporting process: COOLR is media-derived, and in this region roads,
towns and reporters are in the valleys and along the coast.

The consequence was not subtle. Scoring five reference points, the model ranked
the **Columbia Gorge (44° slope, 481 m relief) at 0.040** and a **flat
Willamette valley floor (17°, 53 m) at 0.196** — five times more dangerous.
The model was inverted with respect to landslide physics.

Dropping elevation did **not** fix it, which ruled out "one bad feature" as the
explanation and pointed at the sampling design instead.

## D11 — The actual bug was uniform background sampling

Uniform background over the study box fills the negatives with the Cascades and
Olympics: enormous tracts of steep, remote terrain that nobody reports on. The
model correctly learned what it was shown — *steep and remote = negative* —
which is a fact about observer coverage, not about slopes.

**Fix:** target-group background sampling (`features/sampling.py`). Each
background point is anchored on a randomly chosen real event and offset by up
to 12 km (area-weighted within the disc, so it does not clump at the anchor).
The background then shares the positives' accessibility footprint, observer
effort cancels between the classes, and what is left for the model to learn is
terrain. This is the standard species-distribution-modelling correction, which
the brief already pointed at.

Effect on the model, same features, same CV, same seed:

| | uniform background | target-group |
|---|---|---|
| PR-AUC | 0.7063 | 0.5657 |
| lift over base | 4.24× | 3.39× |
| `elev_m` share of gain | 31% (57% tuned) | 22.7% |
| top-6 gain spread | elevation dominant | elev 23%, tpi 21%, roughness 16%, tri 11%, slope 9%, relief 8% |

**The headline number went down and the model got better.** The 0.71 was
substantially a measurement of where roads are.

Validation that the corrected model is physically sensible — observed rate and
prediction across slope octiles, out-of-fold:

| slope | observed rate | mean predicted |
|---|---|---|
| 0–2° | 0.075 | 0.053 |
| 8–12° | 0.158 | 0.134 |
| 17–23° | 0.193 | 0.163 |
| 30–66° | **0.287** | 0.257 |

Landslide rate rises monotonically with slope (3.8× across the range) and the
model tracks it (corr predicted-vs-observed +0.509). Note that the five-point
spot check is *still* not monotone — individual predictions from a 0.57 PR-AUC
model are noisy, and judging a probabilistic model by a handful of hand-picked
points is not sound. The population relationship is the evidence.

Ablation after the fix (`eval/ablation.py`):

| Feature set | PR-AUC | Lift | vs all |
|---|---|---|---|
| all 9 features | 0.5657 | 3.39× | — |
| no `elev_m` | 0.4763 | 2.86× | −15.8% |
| no `elev_m`, no `tpi` | 0.3850 | 2.31× | −31.9% |
| terrain shape only | 0.3897 | 2.34× | −31.1% |

Residual caveat: elevation and TPI still carry ~32% between them. TPI is partly
legitimate physics (landslides initiate on convergent slopes and deposit in
hollows), so this is not all bias — but elevation's remaining share should be
treated as suspect, and the susceptibility score read as **relative within the
study region**, not as an absolute probability.

Still worth doing:

1. **Road-proximity feature** (OpenStreetMap) — named in the brief. Making
   accessibility explicit lets the model separate it from terrain rather than
   smuggling it through elevation.
2. **Validate against the Myanmar/Vietnam inventories**, which are
   satellite-mapped rather than reported and carry no accessibility bias at all.

The trigger layer is immune to all of this by construction: its controls are
the same location on other dates, so accessibility is constant within a stratum
and cannot contribute.

## D12 — Trigger rainfall came from NASA POWER, not Open-Meteo

Even after the D9 reductions the Open-Meteo pull needed ~3 hourly quota windows.
NASA POWER (`power.larc.nasa.gov`) serves the same shape of data — daily
`PRECTOTCORR`, 1981–present, no key, no practical rate limit — and returned a
21-year point series in **1.5 s**. All 92 cells fetched in under a minute.

Tradeoff: POWER is ~0.5° (MERRA-2) against ERA5-Land's 0.1°. In this region
rainfall varies sharply over 55 km, so a POWER cell smooths orographic
gradients and points 30 km apart can share a series. **This costs resolution,
not validity**: the case-crossover design compares a cell against itself on
other dates, and every feature is a percentile against that cell's own record,
so a coarse cell adds noise to the trigger signal but cannot bias it toward
either class.

The two sources are interchangeable behind one interface
(`build_dataset.py --weather {nasapower,openmeteo}`), and the manifest records
which one produced the matrix so `serve/score.py` scores with the same product
it trained on. Mixing them would feed the model percentiles from a different
product on a different grid — numerically valid, quietly wrong.

Consequence for CV grouping: with 0.5° cells, several 0.25° spatial blocks fall
inside one rainfall series. The trigger layer therefore groups on **`wx_cell`,
not `block_id`** — the weather cell is the real unit of independence, and the
finer group leaked. It mattered: PR-AUC 0.680 under `block_id` vs **0.654**
under `wx_cell`.

## D13 — The combination rule is gated on the trigger, not on "either layer"

The obvious rule — amber if either layer crosses its threshold — is wrong for a
warning product, and the demo output showed why: susceptibility is a property
of the hillside and does not change day to day, so anywhere on steep ground sat
at amber *every day of the year*, including dry July afternoons. A permanent
warning is not a warning; it teaches people to ignore it.

Final rule: **the trigger gates the alert, susceptibility sets its severity.**

| trigger | susceptibility | status |
|---|---|---|
| below threshold | any | green |
| above | below | amber |
| above | above | red |

This is also how LHASA reports hazard. Susceptibility is still published in the
payload as standing context; it just cannot raise an alarm by itself.
Thresholds are each model's recall-tuned operating point (80% recall), not
round numbers.

Verified across three dates:

| date | result |
|---|---|
| 2015-12-08 (major storm) | 4 red, 1 amber — trigger 0.88–1.00 |
| 2016-10-14 (storm) | 4 red, 1 amber — trigger 0.64–0.88 |
| 2015-07-15 (dry) | 5 green — trigger 0.06–0.07 |

## D14 — Prospective hindcast: the base-rate trap, and the model vs a one-line rule

Everything so far was measured on constructed datasets. The hindcast froze the
trigger model on data through 2015 (training, tuning, calibration **and
threshold**), then replayed 2016–2024 over every day at every cell: 302,496
cell-days, 196 reported event-days — deployment base rate **6.5×10⁻⁴**, versus
0.20 in the training set.

**Finding 1 — the threshold trap.** The 80%-recall threshold tuned on the
case-control set alarmed on **39% of all real days**. A threshold only means
something on the distribution it will score; thresholds now come from the
daily grid itself (`serve/calibrate_threshold.py` → `serve/thresholds.json`,
alarm budgets of 2/5/10% of days; the trained threshold 0.138 became 0.402 at
the 5% budget).

**Finding 2 — the honest operating menu** (POD at ±1 day, prospective):

| alarm budget | days/yr/cell | events caught | rain-rule alone |
|---|---|---|---|
| 1% | ~4 | 20% | 13% |
| 2% | ~7 | 22% | 23% |
| 5% | ~18 | 45% | 40% |
| 10% | ~37 | 54% | 55% |

**Finding 3 — the ML barely beats a spreadsheet rule.** The one-line rule
"alarm when `precip_3d_pctl_seasonal` crosses a threshold" matches the
16-feature LightGBM nearly everywhere (the model wins clearly only at the 1%
budget). The skill lives in the *feature framing* — relative-to-own-climatology
— not in the model class. This validates LHASA v1's design (exactly such a
percentile rule) and keeps the trained model only because it is never worse
and calibrates better.

**Finding 4 — a ceiling.** The POD curve flattens near 55–60%: roughly a third
of reported slides fall on unremarkable-rain days (human-triggered, snowmelt,
mis-dated or mis-located reports). No rainfall-only trigger can catch those.

FAR stays >99% at all sane budgets — with a 6.5×10⁻⁴ base rate even a perfect
ranker would. An alarm means "conditions dangerous", not "a slide will be
reported"; every operational warning system carries the same caveat.

## D15 — External check against NASA's global susceptibility map

Sampled Stanley & Kirschbaum's global map (~1 km, classes 1–5, live on the
same gis01 server) at all 5,352 of our points. On identical labels: ours
PR-AUC 0.576 / ROC 0.823; global map 0.258 / 0.667. Our mean score rises
monotonically with their class (0.055 → 0.23 for classes 1→4): the maps agree
about which terrain is dangerous, ours adds 30 m resolution. Stated fairly:
our labels share the reporting process our model was fit to, so this reads as
"sane and locally sharper", not "beats NASA globally" — see D16 for why that
distinction matters.

## D16 — Transfer test: regional terrain models do not travel

The Myanmar (Chin/Rakhine) inventory is satellite-mapped — every visible
failure digitised, no roads-and-reporters filter. The one label source that
cannot share the PNW's observation bias, scored with the PNW-trained terrain
model on 12,000 points (3,000 slide sites, 9,000 target-group background):

| predictor | ROC-AUC |
|---|---|
| PNW model, frozen (transfer) | **0.471** — below chance |
| slope alone (floor) | 0.620 |
| trained on Myanmar (ceiling, same features) | 0.742 |

Four rescue attempts, all failed: drop elevation/TPI (0.484), shape-only
features (0.466), monotone constraints on slope/roughness/relief (0.537),
rank-normalising features within each region (0.522; with constraints 0.544).

Diagnosis: **57–63% of Myanmar terrain lies beyond the PNW's 90th percentile**
in roughness, relief and elevation — most of the country is outside the
model's training support, the trees saturate, and what discrimination remains
comes from regime-specific relationships that do not carry. The features work
there (local model 0.742); the learned *weighting* does not. Notably the PNW
slope response itself is clean — monotone from 0.117 (0–10°) to 0.455
(40–66°) — so this is covariate shift plus regime difference, not simply
leftover reporting bias.

**Consequences.** (a) v1's regional scope was not a limitation to apologise
for — it is the correct scope, now demonstrated rather than assumed. (b) NASA's
choice of a *heuristic* for their global product stops looking conservative
and starts looking wise. (c) The trigger layer, by contrast, passed its own
out-of-distribution test (forward in time, D14) — because its features are
percentiles against local climatology *by construction*. The symmetric lesson:
absolute feature values do not travel, in time or in space.

## D17 — Road distance: the reporting artifact, measured directly

OSM arterial network (motorway→tertiary) for the box via Overpass,
~1.2M+ vertices, nearest-distance via cKDTree on ECEF unit vectors. Median
distance to an arterial road: **positives 23 m, background 1,575 m**. Reported
landslides are literally roadside — partly because slides that block roads get
reported, partly because geocoders snap reports to road features. `road_dist_m`
now rides along in the susceptibility matrix so the model can attribute the
accessibility artifact to accessibility, instead of smuggling it through
elevation.

Retrained results (tuned, spatial CV): PR-AUC **0.839–0.855**, lift ~5.1×,
`road_dist_m` alone carrying 60.7% of gain; elevation's share collapsed from
22.7% to 7.4%. The ablation is the clean part:

| feature set | PR-AUC | vs all |
|---|---|---|
| all 10 (with road) | 0.839 | — |
| no road | 0.566 | −32.6% |
| **no elevation, no TPI (road kept)** | **0.822** | **−2.1%** |
| shape + road only | 0.817 | −2.7% |
| shape only | 0.390 | −53.6% |

Without road, removing elevation+TPI had cost 32%; with road explicit it costs
2%. **Elevation was almost pure road-proxy**, as suspected in D10/D11 and now
demonstrated.

Reading it honestly: the road-aware model predicts *reported/impactful*
landslides — "steep rough ground near roads and people". For a warning product
that is arguably the right target (a slide in trackless wilderness endangers
nobody), and the demo scoring reflects it: Portland's West Hills (its real
chronic slide zone) now scores 0.385 while remote steep terrain scores low.
For land-use planning it would be the wrong target. Two artifacts are kept and
labelled: `susceptibility.pkl` (road-aware, impact-weighted, regional product)
and `susceptibility-terrain-only.pkl` (physics-ish, used by the transfer test).

## D18 — Global scope: tiered confidence, not one global model (user decision)

The user set the target scope to global, explicitly including "warning for
places with possibly low or faulty info". D14/D16 dictate the shape a global
version must take — a naive "train once, score everywhere" model is *proven*
wrong (below-chance transfer), but a tiered system is genuinely buildable:

- **Tier A — regionally modelled.** Where labels support training and
  validation (PNW today; Myanmar could be trained locally at 0.74 tomorrow):
  trained susceptibility + trained trigger, hindcast-verified thresholds.
- **Tier B — global heuristics, real but weaker skill.** Everywhere else:
  NASA's global susceptibility map (or a slope heuristic) for the standing
  layer, and the rain-percentile rule for the trigger. Crucially the trigger
  needs **no labels anywhere** — NASA POWER is global and "3-day rain above
  the 98th percentile for this place and season" is computable for any point
  on Earth — and D14 showed that rule captures most of the trained model's
  skill.
- **Tier C — degraded/unknown inputs.** Where rainfall data quality is poor,
  DEM coverage fails, or nothing is mapped: still render the location, but
  say so — wide uncertainty, no red alerts issued on data that cannot support
  them, an explicit "low confidence" badge instead of silence.

Every scored cell carries its tier; the tier is part of the product, not
metadata. This is also how the "faulty info" requirement is met honestly: the
system warns *and* tells you how much to trust the warning.

Map presentation (decided, not yet built — UI explicitly deferred by user):
**2D map, not a 3D globe.** Landslide risk is consumed at neighbourhood zoom,
where a globe adds nothing but rendering cost and low-bandwidth pain; 2D tiles
work offline-first on cheap phones, which is who this product is for; and
tier/status choropleths read cleanly on a flat projection. The static-JSON
serve layer already matches a tiled map's fetch pattern.

## D19 — "As proficient as possible everywhere": measured, not assumed

The user asked why not aim for maximum proficiency everywhere on Earth. That
is the aim; this experiment measured what actually delivers it. Eight regions
on four continents (7 COOLR satellite inventories + the PNW report catalog,
78,084 rows, terrain-only portable features), evaluated leave-one-region-out —
for each region, every "global" method is tested on a place it has never seen:

| held-out region | local (ceiling) | pooled | pooled-rank | slope | NASA | **ensemble** |
|---|---|---|---|---|---|---|
| Myanmar | 0.742 | 0.611 | 0.662 | 0.621 | 0.559 | 0.645 |
| Vietnam | 0.782 | 0.663 | 0.702 | 0.675 | 0.578 | **0.713** |
| Laos | 0.769 | 0.665 | 0.731 | 0.739 | 0.582 | **0.744** |
| Philippines | 0.676 | 0.618 | 0.649 | 0.551 | 0.543 | 0.601 |
| Brazil | 0.948 | 0.821 | 0.793 | 0.908 | 0.687 | 0.884 |
| Malawi | 0.918 | 0.931 | 0.814 | 0.914 | 0.859 | **0.922** |
| Mexico | 0.577 | 0.587 | 0.564 | 0.584 | 0.568 | **0.588** |
| PNW (reports) | 0.810 | 0.606 | 0.601 | 0.615 | 0.667 | 0.645 |
| **mean / worst** | — | 0.688 / 0.59 | 0.690 / 0.56 | 0.701 / 0.55 | 0.630 / 0.54 | **0.718 / 0.59** |

(ROC-AUC. "ensemble" = equal-weight average of pooled-rank model, slope, and
NASA class, each as a within-region percentile — weights fixed a priori, no
fitting to the held-out region.)

Findings, in order of importance:

1. **Pooling across regimes eliminates the below-chance catastrophe.** The
   single-region export scored 0.47 on Myanmar (D16); the pooled model scores
   0.61–0.66 there and never inverts anywhere. Training support that spans
   the world's terrain regimes is what LHASA 2.0 gets right.
2. **No single global method dominates.** Slope alone wins Brazil (0.908!),
   the rank-pooled model wins Philippines (+0.10 over slope), NASA's map wins
   the reports-regime PNW. Which signal carries a region is itself regional.
3. **The equal-weight ensemble is the right global floor**: best mean (0.718)
   AND best worst-case (0.588) of any method that has never seen the region.
   A floor's job is robustness, and it beats-or-ties the best heuristic in
   6/8 regions without ever being the worst anywhere.
4. **Local models remain far better where labels exist** (gap of +0.08 to
   +0.19 over any global method). Tier A is not optional decoration; it is
   where most of the proficiency lives. All 7 inventory regions now have
   trained, calibrated local artifacts (`models/artifacts/susceptibility-*.pkl`,
   CV ROC 0.58–0.95).
5. Mexico is the honest hard case (~0.58 for everything including local): a
   single-storm inventory in a 0.44°×0.49° box of homogeneous terrain — too
   little contrast to learn from. Some places are Tier B not for lack of
   ambition but because their labels cannot support more yet.

**So the answer to "why not maximum proficiency everywhere" is: this is what
it looks like.** Proficiency is bounded per-place by the labels that exist
there. The architecture maximizes it subject to that bound — ensemble floor
everywhere (never below ~0.59 in any regime tested), local models wherever
labels exist, and every new inventory that appears anywhere on Earth converts
territory from the floor to a local model. The path to raising the floor
further is more *diverse regions in the pool*, not a cleverer model class.

## D20 — Fire danger layer: global by construction, validated on two continents

Built per the brief's framing — danger *conditions*, never ignition prediction.
`pipelines/fireweather.py` computes, from NASA POWER daily weather alone:
KBDI (the brief's named fuel-moisture proxy — a 0–800 running moisture deficit
that self-scales to local mean annual rainfall), vapor pressure deficit, days
since rain, and seasonal percentiles of all of it. **No labels enter the
features**, so the layer is computable for any point on Earth today.

Labels only *validate*. Case-crossover (a fire day vs season-matched non-fire
days at the same location, ±21-day exclusion) against two public databases:
FPA-FOD 6th ed. (USDA, public domain; 40,232 fires ≥100 acres in span) and
Canada's NFDB (open; 9,005 fires ≥100 ha):

| test | ROC-AUC | PR-AUC (base 0.20) |
|---|---|---|
| US, 5-fold CV, 320 cells | 0.759 | 0.424 |
| Canada, 5-fold CV, 220 cells | 0.806 | 0.499 |
| **train US → test Canada cold** | **0.769** | 0.426 |
| train Canada → test US cold | 0.696 | 0.340 |
| pooled CV | 0.780 | 0.453 |
| VPD seasonal percentile alone | 0.72–0.74 | — |

The transfer rows are the finding: fire-weather skill crosses the
temperate→boreal boundary essentially intact (0.769 cold vs 0.806 local) —
the mirror image of D16's terrain result. **Weather-driven layers are global
by construction because percentile features self-normalize; terrain layers
are regional because labels and regimes are.** The architecture's split into
susceptibility (regional) and trigger (global) tiers is now evidenced from
both directions.

Known gaps, stated: 0.5° daily 2 m wind smooths downslope wind events — the
Camp Fire day scores KBDI 691/800 with 35 rainless days but only moderate
model danger, because the 80 km/h Jarbo Gap winds are invisible at this
resolution. And there is no fuel/vegetation layer yet, so barren deserts score
on weather alone. Both are v2 features (ERA5 gusts; a vegetation mask).

## D21 — Drought layer: empirical SPI, 20 years against the US Drought Monitor

The indicator is the same move that carried everything else: a 30/90/180-day
precipitation total as a percentile of the *same calendar window* across all
years — the empirical SPI. Label-free, global, `eval/drought_validate.py`.

Validated against the expert-drawn USDM (open API, county weekly area-%):
~40 counties sampled on a 4° grid across CONUS, 60,573 county-weeks,
2006–2024. Detecting "county ≥50% in severe drought (D2+)":

- **SPI-90 alone: median AUC 0.799** (IQR 0.73–0.87) across counties;
  Spearman with graded severity +0.45.
- Calibrated head (SPI features → P(D2+), county-grouped CV): ROC 0.795,
  PR 0.446 at base 0.161. Saved as `drought_head.pkl`.

Where it underperforms is itself information: precipitation-only SPI misses
snowpack- and temperature-driven drought (the low-IQR counties), and the USDM
blends soil moisture, streamflow and expert judgement — so the ~0.80 ceiling
against it is expected, not a defect. The head's calibration is US-only and
ships with that caveat.

## D22 — Tier-A expansion meets a data-quality wall; Colombia proves the floor

Attempting to add report-based regions: Nepal, NW-India Himalaya and Java
**failed the location-accuracy gate** — after requiring exact/1 km locations
(anything looser makes 30 m terrain features meaningless), only 41–68 usable
sites remained. Media reports in high mountains are geolocated too loosely to
train on. The growth path for Tier A is satellite-mapped inventories, not
news reports — and the places that most need coverage are exactly where
reports are least precise.

Colombia (tropical Andes, 239 exact-located reports) did qualify, and its
LORO debut is the clearest validation of the floor design yet: the pooled
model scored **0.656 on Colombia having never seen it — beating Colombia's
own locally-trained model (0.648) and crushing slope (0.523)**. For
small-label regions, the world pool is better than themselves. 9-region
ensemble floor after the update: mean 0.700, worst-case 0.553.

## D23 — The global scorer: "proficient everywhere" as a runnable contract

`serve/score_global.py` scores ANY (lat, lon, date) on Earth for all three
hazards, each block carrying its tier and caveats:

- **landslide** — Tier A regional artifact when the point falls in one of the
  modeled regions; else the Tier B floor (slope + NASA class + pooled model,
  averaged as percentiles of the global pool). Trigger = rain percentile rule
  (D14: it carries the trained model's skill and needs no labels).
- **fire** — the two-continent-validated danger model; Tier B everywhere.
- **drought** — SPI percentiles + the USDM-calibrated severity head.
- Missing inputs degrade the affected block to **Tier C with the reason
  named** — the user's "warn even with weak info" requirement, honestly.

World demo (all Tier-B points): monsoon Kathmandu reads soaked (KBDI 5,
SPI-90 0.97); Camp Fire day reads KBDI 691, 35 rainless days, drought
P(severe) 0.33; end-of-dry-season Okavango reads KBDI 775 with 175 rainless
days; the Sahara pins KBDI at 797 while the seasonal percentiles correctly
refuse to call a desert being dry "unusual".

## D24 — The case file: ten famous disasters, and what the misses fixed

`eval/case_studies.py` scores ten historical catastrophes through the global
scorer with expectations written down **before** scoring — misses kept
verbatim (`models/runs/case-studies.json`). Outcome: 5 hits (Freetown 2017,
Kedarnath 2013, Fort McMurray 2016, Iowa 2012, Black Saturday borderline at
the alert threshold with VPD 4.94 kPa), 2 misses predicted in advance (Camp
Fire, Marshall Fire — both wind-driven, the documented 0.5° wind gap; in both
the *enabling dryness* was caught), and 3 instructive failures that produced
same-night structural fixes:

1. **Cape Town Day Zero** read near-normal — a 180-day window cannot see three
   failed winters. Fix: **SPI-365** added; Cape Town's annual window now reads
   0.213 vs the 90-day 0.51, and the USDM-validated drought head improved from
   ROC 0.795 to **0.841**.
2. **Horn of Africa 2022** read *wet* during a famine drought — because NASA
   POWER's precipitation is faulty in East Africa (claims 2022 as the wettest
   year on record; ERA5 disagrees by 2× and matches ground truth). Fix:
   `pipelines/precip_quality.py` — a **source-agreement detector** (POWER vs
   ERA5 monthly correlation + annual ratio, cached per cell). Where sources
   disagree, every precip-derived score degrades to **Tier C with the reason
   named**. Discriminates perfectly on test cells: Horn and Cape Town flagged,
   Iowa and the PNW pass. This is the user's "warn even with possibly faulty
   info" requirement made mechanical: the system now knows when its own inputs
   are lying.
3. **Oso 2014** — susceptibility 0.023 and 3-day rain 0.329 at the deadliest
   US landslide. Two mechanisms, one fixed: (a) victims lived on the flat
   valley floor 600 m from the scarp → **neighborhood-max susceptibility**
   (~900 m ring) now reported and gated on; (b) Oso was an antecedent-
   saturation failure — the 30-day rain sat at the **96.2nd percentile** while
   the 3-day was unremarkable → the alert gate now takes the worse of the
   3-day and 30-day percentiles. After both fixes Freetown and Kedarnath raise
   `alert=True` cold; Oso's trigger side gates but its susceptibility stays
   honestly low — a deep-seated failure in glacial outwash is invisible to
   surface slope, and the fix (GLiM lithology / LiDAR scarp morphology) is on
   the roadmap, not tuned around. Refusing to lower the gate until one famous
   anecdote passes is the difference between validation and overfitting.

Also deployment-priced this session: fire alarm thresholds computed on the
real distribution of days (weekly-sampled 541-cell grid; 5% budget threshold
0.4028), same D14 discipline as the landslide trigger.

## D25 — Flood layer: GloFAS integration, gauge-validated, all four hazards live

The user created the ECMWF/EWDS account (the one credential the platform
needed); everything else stayed public/keyless. Exactly per the brief — no
discharge modeling from scratch — `pipelines/glofas.py` consumes
`cems-glofas-historical` v5 (LISFLOOD, 0.05°, daily mean discharge) and
expresses each river cell's flow as a percentile of its own 21-year seasonal
record. River mask: median ≥ 5 m³/s, with channel-snapping to the largest
river cell within ~10 km (a 0.05° channel rarely sits under the queried
point — Oso's lesson applied to rivers).

Integration traps found and fixed, for the record: EWDS request costs are per
*product-type × days* (asking consolidated+intermediate together doubled cost
and tripped the cap — one type per request); the NetCDF longitude arrives on
0–360 (every western-hemisphere query silently missed the grid until
normalized); `open_mfdataset` wants dask (plain per-year concat avoids the
dependency).

Validation against four USGS gauges (NWIS, public, no key), 7,670 shared days
each, spanning three orders of magnitude of river:

| gauge | median m³/s (gauge/GloFAS) | discharge r | percentile ρ | flood-day hit |
|---|---|---|---|---|
| Columbia @ The Dalles | 4,078 / 3,668 | 0.90 | 0.59 | 49% |
| Willamette @ Portland | 589 / 587 | 0.95 | 0.81 | 85% |
| Skagit @ Mount Vernon | 396 / 316 | 0.79 | 0.82 | 64% |
| Nehalem @ Foss (small coastal) | 34 / 29 | 0.91 | 0.84 | 75% |

Flood-day hit = when the gauge sits above its own 98th percentile, GloFAS
also reads ≥95th. The Columbia's 49% is the dam-regulation caveat — The
Dalles is hydropower-scheduled, and no natural-flow model tracks turbines;
free-flowing rivers validate at 64–85%.

Historical replay through the full stack: **Chehalis Dec 2007** (the flood
that closed I-5 for four days) scores the **100.0th** seasonal percentile
(820 m³/s against a 92 m³/s median) and **Nooksack Nov 2021** the **98.8th**
— both auto-alert, both with compound rain+river extreme flags.

Global coverage grows by fixed 6° tiles fetched on demand
(`python pipelines/glofas.py --tile <lat> <lon>`, ~21 queued requests ≈ an
hour per new basin, cached forever). Scoring never blocks on a fetch: an
uncached basin reports Tier C with "no discharge record cached yet", honestly.

**All four of the brief's Tier-1 hazards are now live in the global scorer,
each with its own independent public-data validation.**

## D26 — The whole-world substrate: NASA POWER's public Zarr archive

The user set the target: cover the whole world for every hazard. The blocker
was arithmetic — the point API at ~2s/cell × ~65,000 land cells cannot paint
a planet. The unlock: NASA publishes the **entire POWER daily archive**
(MERRA-2, 0.5°×0.625°, 1981–present, all variables) as an anonymously
readable Zarr store on AWS S3 (`s3://nasa-power/merra2/temporal/...`) — the
same data the point API serves, so every validation done against the API
carries over to bulk reads unchanged.

Architecture built on it:

- `pipelines/power_global.py` streams the store once in latitude bands and
  reduces 25 years of daily history to **fortnightly 21-step quantile
  ladders** per cell for every validated signal (rain3d/30d, spi90/180/365,
  KBDI, VPD, tmax, wind). ~30 GB one-time pass → ~1–2 GB of lookup tables.
- `serve/score_world.py --date D` then scores the planet in one ~400 MB
  trailing-window pull + numpy: global percentile fields and alert masks per
  hazard. `--parity` compares world fields to the point pipelines at
  reference sites (pass bar: mean |Δ| ≤ 0.10) so planetary maps provably say
  what the validations validated. `reports/render_world.py` draws the maps.
- Flood cannot ride this store (discharge lives on EWDS) — its global path
  stays the D25 tile queue, now running a ten-basin priority list ordered by
  flood-exposed population (Ganges first).

Build reliability notes: a rolling-window off-by-one killed the first pass
(fixed with the prefix-sum form, brute-force verified); s3fs's async loop
corrupts after ~20 GB streamed in one Windows process ("Token … different
Context"), so the build runs under an auto-retry wrapper — bands are cached
and each fresh process resumes where the last died.

## D27 — Fire hindcast: the model earns its keep, the threshold did not

*(Workstream executed by a delegated agent under D14's protocol; reviewed and
its threshold finding fixed by the manager the same hour.)*

Frozen on ≤2015 (1,216 strata; 113 dropped for losing every control to the
split), replayed to the end of each label record — US 2016–2020 (FPA-FOD stops
at fire year 2020), Canada 2016–2024 (NFDB runs later; the weather cache is
the binding limit). 539 cells, **1,304,712 cell-days**, 2,087 reported
large-fire days: deployment base rate **1.6×10⁻³**, vs 0.32 in training.

**Finding 1 — fire is where the ML actually pays** (POD ±1 day, prospective):

| alarm budget | days/cell/yr | events caught | VPD rule alone |
|---|---|---|---|
| 2% | 7.9 | 22.3% | 15.3% |
| 5% | 20.9 | 44.9% | 28.9% |
| 10% | 40.9 | 66.3% | 44.0% |

The 14-feature model beats the one-line `vpd_pctl_seasonal` rule by
1.45–1.55× at every budget — the exact inverse of D14, where the rain rule
matched the trained model. Fire danger is genuinely multi-signal (KBDI drought
state + VPD + wind + rainless days); rainfall triggering is not. "The rule is
nearly as good" holds for landslides and does **not** transfer to fire.

**Finding 2 — the recorded thresholds were structurally wrong, now fixed.**
The 5% fire threshold 0.4028 alarmed on 14.0% of days (51/cell/yr). Not
climate drift — the diagnostic reproduced 0.4028 exactly on three sampling
designs. The cause: **0.4028 is exactly 29/72, an isotonic *plateau mean***,
and 3.43% of all days score exactly that value; production serves
`danger >= thr`, so the whole tied block alarms (P(>thr)=4.87%,
P(≥thr)=8.30%). The landslide threshold 0.40196 = 41/102 — same disease.
**Fix (manager):** `tie_safe_threshold()` in both calibrators — choose the
smallest *distinct* score whose ≥-rate fits the budget and record the
**achieved** rate. Recomputed: trigger 0.4040 achieving 4.95%; fire 0.4067
achieving 4.86%. Isotonic steps are coarse, so some budgets undershoot
honestly (landslide "10%" achieves 7.53%) — recorded, not rounded away.

**Finding 3 — one continental threshold quietly taxes the US to subsidise
Canada** (6.94% of US days for 41.1% POD vs 4.71% of Canadian days for 53.9%
at the pooled threshold; boreal fire is more weather-determined, matching
D20's CV gap). Per-region thresholds are the v2 fix.

Data hygiene: NFDB writes missing coordinates as **zeros** — 4 of D20's 2,160
strata sat at (0.0, 0.0), pairing Parks Canada fires with Gulf-of-Guinea
weather. Guards added to both label loaders; effect on D20's numbers is
negligible (4/2160) but the class of bug is now fenced. FAR stays >97% at
every budget; at this base rate it must.

## D28 — Population exposure: alerts denominated in people, not cells

*(Delegated agent workstream; reviewed and wired into the world scorer by the
manager.)*

`pipelines/population.py` gives every hazard field its missing denominator.
Source: GHS-POP R2023A, epoch 2025, 30 arcsec, EPSG:4326 (EU JRC), CC BY 4.0 —
licence read from the product directory's own copyright.txt rather than from
memory and archived beside the raster; "reuse allowed provided credit is given
and changes are indicated" covers redistributing the aggregate. WorldPop stayed
the unused fallback (its 1 km mosaic path 404s); GHSL needs no reprojection.

The file is not the grid one would assume, and each surprise changed the code:
43202 × 21384, origin −180.00792 (POWER edges fall mid-pixel, so pixels are
assigned by centre and the global total stays exact), and a 360.017° span —
two columns MORE than the globe, holding 284 vs 1,456 people over the same
ground. Summing all 43202 double-counts that strip, so the build reads a
contiguous 43200-column window, after which the fold is exact (75 px per POWER
lon cell) and is proved against a brute-force bincount before use.

Global total 8.192B (sanity band 7.5–8.5B); Cairo 24.9M, Mumbai 22.4M,
Kolkata 22.0M, Seoul 21.5M, Delhi 20.5M lead; Sahara and mid-Pacific read 0;
per-cell windowed reads reproduce the built grid bit-exactly. Recorded limits:
GHS-POP under-maps the high Arctic (a sub-Arctic zero means unmapped, not
empty), and epoch 2025 is GHSL's *projected* surface disaggregated from the
2020 observation base — chosen because exposure should answer "who is there
now"; switching to observed E2020 is a one-line change.

`serve/score_world.py` now reports `people` beside `cells` for every alert
mask: "this alert covers 2.3M people" instead of "417 cells flagged".

## D29 — Substrate features: a small honest in-region gain, and worse transfer

*(Delegated agent workstream; negative verdict accepted by the manager —
geology enters no production model. Numbers in
`models/runs/geology-experiment.json`.)*

The D24 roadmap named lithology as the Oso fix. Measured on three regions,
untuned BASE_PARAMS, GroupKFold(5) on block_id. SoilGrids 250m (ISRIC,
CC-BY 4.0) via 216 WCS GeoTIFF windows (99 MB, no point-API hammering); GLiM
(CC-BY 3.0, PANGAEA) at the only openly-archived resolution, 0.5° — the
full-resolution polygons route through a commercial publisher and are
unreachable under the keyless/open constraint.

(a) In-region geology helps a little, and it is all soil texture:

| region | terrain | +soil | +soil+lith | Δ ROC | lith marginal |
|---|---|---|---|---|---|
| pnw | 0.810 | 0.820 | 0.822 | +0.012 | +0.002 |
| myanmar | 0.743 | 0.755 | 0.758 | +0.015 | +0.002 |
| brazil | 0.952 | 0.953 | 0.953 | +0.001 | +0.000 |

Soil carries 15–37% of model gain; 0.5° lithology carries under 1% and adds
≤0.002 ROC over soil alone — a ~55 km cell is a resolution result, not a
geology result.

(b) **Transfer got worse**: pnw+brazil→myanmar 0.515→0.484 (−0.031),
pnw+myanmar→brazil 0.706→0.661 (−0.045), both still under the slope floor.
Mechanism, D16 extended to substrate: 72.1% of Myanmar sits in a lithology
class with zero training examples; 79.1% of Brazil's clay values fall outside
the training 5–95 percentile band. Absolute substrate values travel no better
than absolute terrain values.

(c) **Oso: no, exactly as pre-registered.** SoilGrids reports the top 30 cm as
sandy and well-drained — which reads as *stable* — while 2014's failure was
deep-seated in lacustrine clay beneath glacial outwash, below the mapped
horizon; GLiM calls the cell metamorphic Cascades core, not valley fill. The
point score rose (0.181→0.251) but the neighbourhood-max the product gates on
**fell 0.333→0.251, below the 0.30 gate: adding geology would have un-flagged
Oso.** The honest Oso fix remains LiDAR-derived scarp morphology or a mapped
deep-seated-landslide inventory, not any open global soil/lithology raster.

Caveat limiting even (a): PNW SoilGrids coverage is 80.3% and its missingness
tracks the D10/D11 coastal reporting artifact, so part of PNW's +0.012 is that
artifact; Myanmar (100% coverage) is the trustworthy line.

---

## Open / not yet done

- **AlphaEarth embeddings.** Needs Earth Engine auth (Google Cloud project +
  OAuth), which is the user's to grant. The distance-from-own-history feature
  is specified in the brief and is *not* in v1's feature set. Everything else
  is built so it can slot in as one more susceptibility column.
- **Soil / lithology.** SoilGrids (ISRIC) and GLiM are reachable without auth;
  not yet wired in.
- **Benchmark against LHASA.** `LHASA_Hazard_*` services are live on the same
  server; scoring our model against the operational one is the honest test.
- **Combination rule.** Brief says start with the LHASA-style "both layers
  cross a threshold" rule before any joint model.
