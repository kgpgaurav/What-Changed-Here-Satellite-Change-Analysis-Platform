"""Load and validate the YAML config into typed, immutable objects.

Keeping all run parameters in one validated object means every other module
receives plain values and never parses YAML itself.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import yaml

from wch.geo import bbox_size_km

log = logging.getLogger(__name__)

MAX_AOI_SIDE_KM = 30  # keep runs laptop-sized (8 GB RAM)
REQUIRED_BANDS = {"red", "nir", "green", "blue", "swir16"}  # swir16 (B11) for NDBI/MNDWI
METHODS = {"median", "single"}
# Bump when a code change alters composite results (e.g. a new date-selection rule),
# so cached composites made by older code are rebuilt automatically.
PIPELINE_VERSION = "3.0"  # 2.1: dates spread across window; 3.0: + SWIR band


@dataclass(frozen=True)
class AOI:
    name: str
    bbox: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


@dataclass(frozen=True)
class Period:
    label: str
    start: date
    end: date

    @property
    def stac_datetime(self) -> str:
        """STAC APIs take a date range as 'start/end'."""
        return f"{self.start.isoformat()}/{self.end.isoformat()}"


@dataclass(frozen=True)
class Compositing:
    method: str          # "median": cloud-masked median of many dates; "single": one season-matched date
    max_scenes: int      # at most this many (clearest) dates per period
    min_clear_obs: int   # a pixel needs at least this many cloud-free dates to be kept

    @property
    def effective_min_clear_obs(self) -> int:
        return 1 if self.method == "single" else self.min_clear_obs


@dataclass(frozen=True)
class LandCover:
    """Thresholds for the simple rule-based land-cover classes (see landcover.py)."""
    water_mndwi: float      # MNDWI above this -> water
    vegetation_ndvi: float  # NDVI at or above this -> vegetation
    bare_ndvi: float        # NDVI below this -> built-up/bare; in between -> sparse vegetation


@dataclass(frozen=True)
class TimeSeries:
    """Monthly NDVI time series at coarse resolution (see timeseries.py)."""
    start_year: int
    end_year: int
    resolution: int                  # metres; coarse keeps 8 years of data laptop-sized
    scenes_per_month: int
    max_cloud_cover: float
    min_zone_clear_fraction: float   # a zone's value on a date needs this much cloud-free area
    anomaly_z: float                 # robust z-score above this flags a month as unusual


@dataclass(frozen=True)
class Labels:
    """Which points to label for the classifier (see sampling.py)."""
    labels_dir: Path                 # labels are project data: kept outside data/ so they can be committed
    n_points: int
    min_spacing_m: float
    strata_shares: dict              # stratum name -> share of the points
    seed: int


@dataclass(frozen=True)
class Model:
    """Random forest settings and evaluation (see modeling.py, evaluate.py)."""
    n_estimators: int
    block_size_m: float              # spatial cross-validation block size
    cv_folds: int
    probability_threshold: float     # probability at or above this = "construction"
    seed: int


@dataclass(frozen=True)
class Config:
    aoi: AOI
    before: Period
    after: Period
    stac_url: str
    collections: tuple[str, ...]
    max_cloud_cover: float
    pair_cloud_limit: float
    bands: tuple[str, ...]
    resolution: int
    compositing: Compositing
    landcover: LandCover
    timeseries: TimeSeries
    labels: Labels
    model: Model
    change_threshold: float
    output_dir: Path
    catalog_dir: Path

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.aoi.name

    @property
    def periods(self) -> tuple[Period, Period]:
        return (self.before, self.after)

    def input_fingerprint(self) -> str:
        """Short hash of every setting that affects the composites.

        If any of these (or the pipeline version) change, cached composites are
        stale and get rebuilt.
        Settings that only affect later steps (e.g. change_threshold) are excluded.
        """
        key = {
            "pipeline_version": PIPELINE_VERSION,
            "bbox": self.aoi.bbox,
            "windows": [p.stac_datetime for p in self.periods],
            "collections": self.collections,
            "max_cloud_cover": self.max_cloud_cover,
            "pair_cloud_limit": self.pair_cloud_limit,
            "bands": self.bands,
            "resolution": self.resolution,
            "compositing": asdict(self.compositing),
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]

    def timeseries_fingerprint(self) -> str:
        """Like input_fingerprint, for the downloaded time-series data."""
        ts = self.timeseries
        key = {
            "pipeline_version": PIPELINE_VERSION,
            "bbox": self.aoi.bbox,
            "collections": self.collections,
            "years": [ts.start_year, ts.end_year],
            "resolution": ts.resolution,
            "scenes_per_month": ts.scenes_per_month,
            "max_cloud_cover": ts.max_cloud_cover,
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def validate_bbox(bbox) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError(f"bbox is not a valid lon/lat box: {bbox}")
    return (min_lon, min_lat, max_lon, max_lat)


def _period(label: str, raw: dict) -> Period:
    # YAML already turns 2019-02-01 into a date; str() handles both date and string input.
    start = date.fromisoformat(str(raw["start"]))
    end = date.fromisoformat(str(raw["end"]))
    if start >= end:
        raise ValueError(f"period '{label}': start must be before end")
    return Period(label, start, end)


def _compositing(raw: dict) -> Compositing:
    comp = Compositing(
        method=str(raw.get("method", "median")),
        max_scenes=int(raw.get("max_scenes", 8)),
        min_clear_obs=int(raw.get("min_clear_obs", 2)),
    )
    if comp.method not in METHODS:
        raise ValueError(f"compositing.method must be one of {sorted(METHODS)}")
    if comp.max_scenes < 1 or comp.min_clear_obs < 1:
        raise ValueError("max_scenes and min_clear_obs must be >= 1")
    if comp.min_clear_obs > comp.max_scenes:
        raise ValueError("min_clear_obs cannot exceed max_scenes")
    return comp


def _landcover(raw: dict) -> LandCover:
    rules = LandCover(
        water_mndwi=float(raw.get("water_mndwi", 0.0)),
        vegetation_ndvi=float(raw.get("vegetation_ndvi", 0.4)),
        bare_ndvi=float(raw.get("bare_ndvi", 0.2)),
    )
    if not rules.bare_ndvi < rules.vegetation_ndvi:
        raise ValueError("landcover: bare_ndvi must be below vegetation_ndvi")
    return rules


def _timeseries(raw: dict, before: Period, after: Period) -> TimeSeries:
    ts = TimeSeries(
        start_year=int(raw.get("start_year", before.start.year)),
        end_year=int(raw.get("end_year", after.end.year)),
        resolution=int(raw.get("resolution", 100)),
        scenes_per_month=int(raw.get("scenes_per_month", 2)),
        max_cloud_cover=float(raw.get("max_cloud_cover", 30)),
        min_zone_clear_fraction=float(raw.get("min_zone_clear_fraction", 0.5)),
        anomaly_z=float(raw.get("anomaly_z", 3.5)),
    )
    if ts.start_year > ts.end_year:
        raise ValueError("timeseries.start_year must not be after end_year")
    if ts.end_year - ts.start_year > 10:
        raise ValueError("timeseries: keep the range to at most ~10 years on a laptop")
    return ts


DEFAULT_STRATA_SHARES = {"veg_to_built": 0.35, "sparse_to_built": 0.20, "other_change": 0.20, "stable": 0.25}


def _labels(raw: dict) -> Labels:
    shares = {k: float(v) for k, v in raw.get("strata", DEFAULT_STRATA_SHARES).items()}
    unknown = set(shares) - set(DEFAULT_STRATA_SHARES)
    if unknown:
        raise ValueError(f"labels.strata: unknown strata {sorted(unknown)}")
    labels = Labels(
        labels_dir=Path(raw.get("dir", "labels")),
        n_points=int(raw.get("n_points", 300)),
        min_spacing_m=float(raw.get("min_spacing_m", 150)),
        strata_shares=shares,
        seed=int(raw.get("seed", 42)),
    )
    if labels.n_points < 20 or any(v < 0 for v in shares.values()) or sum(shares.values()) <= 0:
        raise ValueError("labels: need n_points >= 20 and non-negative strata shares")
    return labels


def _model(raw: dict) -> Model:
    model = Model(
        n_estimators=int(raw.get("n_estimators", 200)),
        block_size_m=float(raw.get("block_size_m", 2000)),
        cv_folds=int(raw.get("cv_folds", 5)),
        probability_threshold=float(raw.get("probability_threshold", 0.5)),
        seed=int(raw.get("seed", 42)),
    )
    if model.cv_folds < 2 or not 0 < model.probability_threshold < 1:
        raise ValueError("model: cv_folds must be >= 2 and probability_threshold in (0, 1)")
    return model


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    bbox = validate_bbox(raw["aoi"]["bbox"])
    width, height = bbox_size_km(bbox)
    if max(width, height) > MAX_AOI_SIDE_KM:
        raise ValueError(
            f"AOI is {width:.0f} x {height:.0f} km; keep each side under {MAX_AOI_SIDE_KM} km"
        )

    before = _period("before", raw["periods"]["before"])
    after = _period("after", raw["periods"]["after"])
    if before.end >= after.start:
        raise ValueError("'before' period must end before 'after' period starts")

    bands = tuple(raw["processing"]["bands"])
    missing = REQUIRED_BANDS - set(bands)
    if missing:
        raise ValueError(f"config is missing required bands: {sorted(missing)}")

    return Config(
        aoi=AOI(name=raw["aoi"]["name"], bbox=bbox),
        before=before,
        after=after,
        stac_url=raw["stac"]["url"],
        collections=tuple(raw["stac"]["collections"]),
        max_cloud_cover=float(raw["stac"]["max_cloud_cover"]),
        pair_cloud_limit=float(raw["stac"].get("pair_cloud_limit", 5)),
        bands=bands,
        resolution=int(raw["processing"]["resolution"]),
        compositing=_compositing(raw.get("compositing", {})),
        landcover=_landcover(raw.get("landcover", {})),
        timeseries=_timeseries(raw.get("timeseries", {}), before, after),
        labels=_labels(raw.get("labels", {})),
        model=_model(raw.get("model", {})),
        change_threshold=float(raw["processing"]["change_threshold"]),
        output_dir=Path(raw.get("output_dir", "data/processed")),
        catalog_dir=Path(raw.get("catalog_dir", "data/catalog")),
    )