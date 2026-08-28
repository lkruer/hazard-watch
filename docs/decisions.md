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
