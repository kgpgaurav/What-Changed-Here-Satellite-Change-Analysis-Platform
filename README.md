# What Changed Here?

Select an area and two time periods; the pipeline fetches public Sentinel-2
imagery, classifies simple land cover in each period, maps what changed into what,
and shows how the area behaved month by month in between.

> **Status: Stage 4 (construction classifier with hand labels and spatial evaluation).** The full README (architecture,
> assumptions, limitations, evaluation) is written in Stage 5.

## Setup (Windows, pip)

Python 3.11 or 3.12 recommended. All dependencies, including rasterio/GDAL, install as wheels from PyPI.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest                      # offline tests, no network needed
```

## Run

```powershell
wch search --config config/kanpur.yaml   # candidate dates per period (metadata only, seconds)
wch run    --config config/kanpur.yaml   # build composites (first run ~5-15 min), compute change
wch run    --config config/kanpur.yaml --force   # rebuild composites even if cached
wch timeseries --config config/kanpur.yaml       # monthly NDVI 2019-2026 per zone (after `run`)
wch sample     --config config/kanpur.yaml       # choose ~300 points to label (after `run`)
streamlit run app/streamlit_app.py       # Map / What changed / Over time / Label / Model
wch train      --config config/kanpur.yaml       # train + evaluate the classifier (after labelling)
```

Outputs go to `data/processed/<aoi name>/`:

| File | Content |
|---|---|
| `before_rgb.tif`, `after_rgb.tif` | 8-bit true-colour images (display only) |
| `before_ndvi.tif`, `after_ndvi.tif` | NDVI, float32, NaN = no data |
| `ndvi_diff.tif`, `ndbi_diff.tif` | NDVI and built-up index (NDBI) after − before |
| `before_landcover.tif`, `after_landcover.tif` | 1 water, 2 built-up/bare, 3 vegetation, 4 sparse vegetation |
| `change_type.tif` | 1 stable veg, 2 stable built, 3 stable water, 4 veg → built, 5 built → veg, 6 water change, 7 sparse |
| `composites/<period>_reflectance.tif` | cloud-masked median composite (blue, green, red, nir, swir16) |
| `composites/<period>_clear_obs.tif` | number of cloud-free dates behind each pixel |
| `summary.json` | dates used, offset rules, data coverage, statistics, class areas, transitions |
| `timeseries/monthly.csv` | monthly NDVI per zone with anomaly flags |
| `timeseries/season_by_year.csv` | analysis-month NDVI per zone per year |
| `model/metrics.json` | baselines vs. random forest, confusion matrix, feature importance, area estimates |
| `model/construction_probability.tif`, `model/construction.tif` | model output for every pixel |

Labels live in `labels/` (not in `data/`), because they are hand-made project data that
should be kept with the code: `<aoi>_points.csv` (which pixels) and `<aoi>_labels.csv`
(your answers: 1 construction, 0 no construction, -1 unsure).

`data/catalog/` is a STAC catalog describing every output: source scenes, dates and a
fingerprint of the settings. If the fingerprint matches on the next run, the composite
is reused instead of re-downloaded.

## Project layout

```
config/            one YAML per study area: bbox, date windows, thresholds
src/wch/
  config.py        load + validate config
  geo.py           UTM zone, bbox helpers
  search.py        STAC search, choose which date(s) to use       [network]
  load.py          read pixels for our bbox onto a fixed grid       [network]
  radiometry.py    raw values -> reflectance, 2022 offset handling  [pure]
  composite.py     SCL cloud mask + per-pixel median composite      [pure]
  catalog.py       local STAC catalog: provenance + caching
  indices.py       NDVI, NDBI (built-up), MNDWI (water)             [pure]
  change.py        NDVI difference map + statistics                 [pure]
  landcover.py     rule-based classes, change types, transitions    [pure]
  timeseries.py    zone weights, monthly series, robust anomalies   [pure]
  sampling.py      strata + spaced stratified sampling of points    [pure]
  features.py      14 per-pixel features for the classifier         [pure]
  evaluate.py      spatial CV, metrics, stratified area estimate    [pure]
  modeling.py      `wch sample` / `wch train`, label files
  display.py       colouring for maps                               [pure]
  io.py            GeoTIFF / JSON writing
  pipeline.py      wires the steps together
  cli.py           `wch` command
app/               Streamlit viewer (reads outputs only)
tests/             offline tests on synthetic rasters, incl. a full pipeline run with fake downloads
```

## Land-cover rules

```
MNDWI > 0      -> water
NDVI >= 0.4    -> vegetation
NDVI <  0.2    -> built-up / bare
otherwise      -> sparse vegetation (buffer: not counted as change)
```
Thresholds live in the config and can be changed without re-downloading.

## Machine-learning component (Stage 4)

- **Task:** per 10 m pixel, did new construction appear (building, road, paved or cleared site)?
- **Labels:** ~300 points, drawn per stratum because construction is rare (~1% of the area):
  vegetation → built-up (35%), sparse → built-up (20%), other change (20%), no change (25%),
  at least 150 m apart. Labelled blind in the app (the labeller never sees the stratum).
- **Features:** NDVI, NDBI, MNDWI and brightness before/after/difference, plus the mean
  NDVI and NDBI change in the surrounding 50 m.
- **Model:** random forest (200 trees, balanced class weights).
- **Baselines:** NDVI drop > 0.2 (Stages 1-2) and the vegetation → built-up/bare rule (Stage 3).
- **Evaluation:** spatial cross-validation with 2 km blocks (a block is never split between
  training and test); a random split is reported alongside to show its optimism.
- **Area:** estimated from the labels with a stratified estimator (stratum area × share of
  construction among its labels), with a 95% interval; the model's own map area is shown next to it.

## Known limitations (so far)

- Sentinel-2's cloud classifier (SCL) is imperfect: thin haze can slip through, and
  bright roofs are sometimes flagged as cloud (visible as low clear-observation counts).
  The median limits the damage; pixels with fewer than 2 clear dates are left empty.
- "Built-up/bare" mixes buildings with bare soil, fallow fields and river sand; NDBI
  cannot separate them. Stage 4 trains a classifier for that.
- Fixed thresholds suit Feb-Mar in the Gangetic plain; other seasons/regions need tuning.
- The Ganga's shifting sandbars and channels show up as "water change".
- The labelled set oversamples likely change, so precision/recall describe that set, not a
  random pixel. The stratified area estimate corrects for this; per-pixel scores do not.
- Labels come from one person looking at 10 m imagery plus current high-res imagery of
  unknown date: some answers will be wrong, and there is no second labeller to measure that.
- A stratum with zero labelled construction gets zero variance in the interval, which
  understates uncertainty for rare strata.
- Compositing: set `compositing.method: single` in the config to reproduce the
  Stage 1 single-date approach for comparison.