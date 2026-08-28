# Multi-Hazard Risk Platform — Project Brief

## Vision

A free, public, location-based hazard risk tool. A user picks or is given a
location and gets back current risk scores for the natural hazards we can
reliably model there. Built on Google's AlphaEarth Foundations satellite
embeddings (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` in Earth Engine, verify
current asset name in the EE Data Catalog before first use, these are
versioned) as a shared land-cover/land-change signal, combined with
hazard-specific public datasets.

No accounts, no login, no push notifications. Static site, pull-based. A
user reaches their result by URL (see Delivery section). This is a
deliberate scope decision, not a placeholder: there is no free-at-scale push
channel (SMS/WhatsApp both cost money past a small volume), so the product
is designed around that constraint instead of fighting it.

## Scope for v1 — read this before writing any hazard-specific code

We model hazards where real predictive skill exists and is documented in the
literature. We do NOT attempt to predict earthquakes or tsunamis (no known
precursor signal exists for either; building a "prediction" for these would
produce a confident-looking number with no real skill). If those are
represented at all, it is as exposure/vulnerability maps and a relay of
official warnings (USGS earthquake feed, NOAA/PTWC tsunami warnings), never
as a trained model output. Do not build a classifier for either.

Build order (each hazard reuses the shared pipeline below, only the label
source and a few features change):

1. **Landslide** — build this one completely first. Cleanest label source
   (event catalog with lat/lon/date), clearest existing reference
   implementation to study (NASA's LHASA 2.0, open source:
   github.com/nasa/LHASA). Get the full pipeline (ingest → features → model
   → spatial CV → eval → serve → site) working end to end on this hazard
   before generalizing.
2. **Fire** — reuses almost the entire pipeline. Note: label this as fire
   *danger conditions*, not "a fire will start here." Most fires are
   human-ignited (arson, power lines, equipment), which is not predictable
   from environmental data. What's real and modelable is the conditions
   that make a fire dangerous if one starts, which is what official fire
   danger indices (US NFDRS, Canadian FWI) already do.
3. **Drought / vegetation stress** — slowest-onset, longest lead time,
   arguably the most reliable of the three. Good candidate to add
   AlphaEarth as the primary signal rather than a secondary one.
4. **Flood (river)** — do not build discharge forecasting from scratch.
   Consume GloFAS (Global Flood Awareness System, free, EU/ECMWF) output and
   layer AlphaEarth land-cover/imperviousness change on top for hyperlocal
   exposure. This is an integration task, not a modeling task.
5. Stretch / later: avalanche, land subsidence (InSAR-based, structurally
   different, trend-extrapolation not event-classification), locust/pest
   risk (FAO Desert Locust Watch precedent).

## Shared pipeline architecture

Every Tier-1 hazard uses the same two-layer design:

- **Susceptibility layer** (slow-changing): intrinsic vulnerability of a
  location. Built from terrain, soil, geology, land cover. Retrained
  infrequently, refreshed whenever the slowest input updates (AlphaEarth =
  yearly).
- **Trigger layer** (fast-changing): the condition that pushes a susceptible
  location into an actual event. Usually rainfall, temperature, wind, or
  snowmelt. Refreshed daily.
- Combined score: either a simple rule (both layers must cross a threshold,
  this is literally how LHASA works) or a single model trained jointly on
  both feature types. Start with the simple rule for v1, it's easier to
  debug and explain, move to a joint model once the pipeline is proven.

## Data sources

Verify every asset ID/version below in its current catalog before use.
Satellite data catalogs get renamed and versioned; do not assume these are
still current by the time this is read.

| Hazard | Labels (free) | Key features | Access |
|---|---|---|---|
| Landslide | COOLR / Global Landslide Catalog (NASA, point data w/ lat/lon/date) | Slope + terrain roughness (SRTM `USGS/SRTMGL1_003` or Copernicus DEM GLO-30), soil (SoilGrids/ISRIC), lithology (Global Lithological Map), antecedent rainfall 3/7/30-day (CHIRPS `UCSB-CHG/CHIRPS/DAILY` or GPM IMERG), AlphaEarth land-cover-change as susceptibility factor, road proximity (OpenStreetMap) | COOLR via data.nasa.gov / Landslide Viewer API. Rest via Earth Engine. |
| Fire (danger, not ignition) | FIRMS historical archive, MTBS burn perimeters (US) | Fuel moisture proxy (Keetch-Byram Drought Index, computable from precip deficit + temp), wind speed, AlphaEarth vegetation/dryness trend | FIRMS API/archive (nasa.gov), weather from ERA5-Land |
| Drought/veg stress | US Drought Monitor, USDA NASS yield records | Vegetation health anomaly vs. seasonal baseline (AlphaEarth), soil moisture anomaly (SMAP or ERA5-Land), cumulative rainfall deficit (CHIRPS) | Public downloads + Earth Engine |
| Flood (river) | Global Runoff Data Centre, USGS gauges (validation only) | Consume GloFAS discharge forecast directly, add AlphaEarth imperviousness/land-cover trend for exposure | GloFAS via Copernicus Climate Data Store (CDS) API |

AlphaEarth-derived feature used across all of the above: cosine or
Mahalanobis distance of a location's current-year embedding vector from its
own multi-year historical mean. This is the "how much has this place's
surface character shifted" signal, computed once per Earth Engine query, not
hazard-specific.

## Modeling approach

- **Framing**: binary classification, event occurred / did not occur, in a
  grid cell over a time window (e.g., 1km, 1 day). Output a calibrated
  probability, not a point prediction. This mirrors how a weather forecast
  gives "70% chance," not a guarantee, and it's the honest bar these models
  can actually hit.
- **Model**: gradient boosted trees (LightGBM or XGBoost), not deep
  learning. These problems have a few dozen meaningful features, not raw
  imagery at scale, and tree ensembles outperform neural nets here while
  training faster and giving interpretable feature importance.
- **Negative sampling**: hazard catalogs are presence-only (no confirmed
  "nothing happened here" records). Sample background points from similar
  terrain/region/time without a recorded event. This is the same problem
  ecological niche modeling solves (MaxEnt, species distribution modeling)
  — worth pulling technique from that literature directly rather than
  reinventing it.
- **Validation — do this correctly from the start**:
  - Spatial block cross-validation. Partition the study area into blocks
    and hold out whole blocks, never a random point split. Nearby points
    have near-identical features and outcomes, a random split leaks
    information and makes results look far better than they are. Use
    `GroupKFold` with a block ID as the group, not `KFold`.
  - Metric: PR-AUC as the primary number, not accuracy (these are rare-event
    problems, a model that always predicts "no event" scores >99% accuracy
    and is worthless). Report a calibration/reliability curve too, since the
    output is used as a probability, not just a ranking.
  - Alert threshold: tune for recall over precision. Missing a real event
    is worse than a false alarm, same tradeoff every real warning system
    accepts.
- **Trigger features**: always relative to that location's own climatology,
  never an absolute threshold. LHASA compares 7-day rainfall against the
  long-term record for that specific pixel, since "heavy rain" means
  something different in Seattle than in the Atacama. Build rolling-window
  features (3/7/30-day cumulative and anomaly-vs-climatology) for anything
  rainfall or temperature driven.

## Repo structure

```
/pipelines/     ingestion scripts per data source (earth_engine.py, coolr.py,
                firms.py, glofas.py, ...), one file per source, not per hazard
/features/      feature engineering: rolling windows, climatology anomalies,
                terrain derivatives, the shared AlphaEarth distance-from-
                history function
/models/        training scripts per hazard, shared base class for the
                two-layer susceptibility/trigger design
/eval/          spatial CV harness, PR-AUC + calibration reporting, shared
                across hazards
/serve/         batch scoring job, writes one small JSON per location/cell
                for the static site to fetch, no live per-request compute
/site/          static frontend, no build step heavier than necessary,
                fetches from /serve output, PWA-capable (cache last-known
                status for offline/low-signal use)
/docs/          this file, plus architecture notes as decisions get made
```

## Delivery (site)

- Static hosting (Cloudflare Pages or GitHub Pages), free regardless of
  traffic.
- URL pattern: `/f/{location_id}` for a single-location report,
  `/dashboard/{org_id}` for a multi-location view (useful for an extension
  worker or emergency manager watching many points at once).
- Registration: tap a location on a map, no typing, no account. Generate a
  QR code at registration time for physical distribution (printed card,
  handed out by whoever already has in-person contact with the end user).
- Backend writes a flat JSON per location on a cron (matches source data
  refresh cadence: daily for trigger layers, yearly for AlphaEarth
  refresh). Frontend just fetches static JSON, no server-side rendering
  needed.
- Status page: one dominant color (green/yellow/red) + one plain-language
  sentence + optional expandable trend chart. Add a "listen" button using
  the browser's built-in Web Speech API (free, no server cost) for
  low-literacy accessibility.

## Explicit non-goals for v1

- No tsunami or earthquake prediction model, ever. Exposure mapping and
  official-warning relay only, and only if it's added at all.
- No SMS/WhatsApp push infrastructure. Pull-based site only.
- No user accounts or login.
- No sub-daily updates beyond what the underlying source data actually
  supports, don't fake precision the inputs don't have.
- No deep learning as a default choice. Justify it explicitly if you reach
  for it over gradient boosted trees.

## First task for Claude Code

Scaffold the repo structure above, then build M0: pull one year of
AlphaEarth embeddings for a small bounding box (pick a real landslide-prone
region so labels exist there), pull COOLR landslide records for the same
box, confirm the spatial join works, and produce one working feature
dataframe. Don't touch modeling until that data pipeline is verified end to
end.
