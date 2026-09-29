"""Stage 4 orchestration: choose points to label, then train and evaluate the classifier.

    wch sample -> labels/<aoi>_points.csv        (which pixels to label)
    app "Label" tab -> labels/<aoi>_labels.csv   (your answers, saved per click)
    wch train  -> data/processed/<aoi>/model/    (metrics, probability map, model file)

Task: for each 10 m pixel, did NEW CONSTRUCTION (building, road, paved or
cleared construction site) appear between the two periods? 1 = yes, 0 = no.
The computations live in sampling.py, features.py and evaluate.py.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.warp import transform as warp_transform
from sklearn.ensemble import RandomForestClassifier

from wch import catalog
from wch.config import Config
from wch.evaluate import (
    classification_metrics, confusion, out_of_fold_proba, spatial_blocks, stratified_area_estimate,
)
from wch.features import FEATURE_NAMES, build_features, iter_row_blocks, sample_matrix
from wch.geo import bbox_center, utm_epsg
from wch.io import write_json, write_raster
from wch.landcover import change_type, classify
from wch.pipeline import get_composites
from wch.sampling import STRATA, allocate, sample_points, strata_map

log = logging.getLogger(__name__)

MIN_LABELS = 20

# --- label files (also used by the app) ---------------------------------------------

def points_path(labels_dir: Path, aoi_name: str) -> Path:
    return Path(labels_dir) / f"{aoi_name}_points.csv"


def labels_path(labels_dir: Path, aoi_name: str) -> Path:
    return Path(labels_dir) / f"{aoi_name}_labels.csv"


def read_labels(path: Path) -> pd.DataFrame:
    """Latest answer per point: label 1 = construction, 0 = no construction, -1 = unsure."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["id", "label", "labelled_at"])
    return pd.read_csv(path).drop_duplicates("id", keep="last")


def append_label(path: Path, point_id: str, label: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if new:
            f.write("id,label,labelled_at\n")
        f.write(f"{point_id},{int(label)},{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")


def undo_last_label(path: Path) -> None:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        df.iloc[:-1].to_csv(path, index=False)

# --- shared layers -------------------------------------------------------------------

def _analysis_layers(cfg: Config):
    """Composites (from cache) and the rule-based land-cover change, recomputed with the
    current config so sampling and training always use the same maps."""
    epsg = utm_epsg(*bbox_center(cfg.aoi.bbox))
    comps = get_composites(cfg, epsg)
    before, after = comps["before"][0], comps["after"][0]
    lc_before, lc_after = classify(before, cfg.landcover), classify(after, cfg.landcover)
    return before, after, lc_before, lc_after, change_type(lc_before, lc_after), epsg

# --- wch sample -------------------------------------------------------------------------

def run_sample(cfg: Config, force: bool = False) -> dict:
    pts = points_path(cfg.labels.labels_dir, cfg.aoi.name)
    labs = labels_path(cfg.labels.labels_dir, cfg.aoi.name)
    if pts.exists() and not force:
        raise RuntimeError(f"{pts} already exists. Re-sampling would make your labels meaningless; "
                           "use --force only if you really want new points.")
    if force and labs.exists():
        backup = labs.with_name(f"{labs.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
        labs.rename(backup)
        log.warning("existing labels moved to %s", backup)

    before, _, lc_before, lc_after, changes, epsg = _analysis_layers(cfg)
    strata = strata_map(lc_before, lc_after, changes)
    counts = allocate(cfg.labels.n_points, cfg.labels.strata_shares)
    rng = np.random.default_rng(cfg.labels.seed)
    df = sample_points(strata.values, before["x"].values, before["y"].values,
                       counts, cfg.labels.min_spacing_m, rng)
    lon, lat = warp_transform(f"EPSG:{epsg}", "EPSG:4326", df["x"].tolist(), df["y"].tolist())
    df["lon"], df["lat"] = np.round(lon, 6), np.round(lat, 6)
    df.insert(0, "id", [f"p{i:04d}" for i in range(1, len(df) + 1)])

    pts.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(pts, index=False)
    per_stratum = df["stratum"].value_counts().to_dict()
    short = {k: v - per_stratum.get(k, 0) for k, v in counts.items() if per_stratum.get(k, 0) < v}
    if short:
        log.warning("some strata had too few spaced candidates: %s", short)
    log.info("wrote %d points to %s", len(df), pts)
    return {"file": str(pts), "points": len(df), "per_stratum": per_stratum}

# --- wch train ----------------------------------------------------------------------------

def _labelled_points(cfg: Config) -> tuple[pd.DataFrame, int]:
    pts = points_path(cfg.labels.labels_dir, cfg.aoi.name)
    if not pts.exists():
        raise RuntimeError("No points yet: run `wch sample --config ...` and label them in the app.")
    labels = read_labels(labels_path(cfg.labels.labels_dir, cfg.aoi.name))
    df = pd.read_csv(pts).merge(labels[["id", "label"]], on="id")
    n_unsure = int((df["label"] == -1).sum())
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if len(df) < MIN_LABELS or df["label"].nunique() < 2:
        raise RuntimeError(f"Need at least {MIN_LABELS} labels including both answers; "
                           f"have {len(df)} ({int(df['label'].sum())} construction).")
    return df, n_unsure


def _make_model(cfg: Config):
    def make():
        # class_weight="balanced": construction is the minority class, so its mistakes count more.
        return RandomForestClassifier(
            n_estimators=cfg.model.n_estimators, min_samples_leaf=2, class_weight="balanced",
            random_state=cfg.model.seed, n_jobs=-1,
        )
    return make


def predict_map(model, features: xr.Dataset) -> xr.DataArray:
    """Construction probability for every pixel (NaN where any feature is missing)."""
    ny, nx = features.sizes["y"], features.sizes["x"]
    proba = np.full((ny, nx), np.nan, dtype="float32")
    for r0, r1, X in iter_row_blocks(features):
        valid = np.isfinite(X).all(axis=1)
        block = np.full(len(X), np.nan, dtype="float32")
        if valid.any():
            block[valid] = model.predict_proba(X[valid])[:, 1]
        proba[r0:r1] = block.reshape(r1 - r0, nx)
    return xr.DataArray(proba, dims=("y", "x"), coords={"y": features["y"], "x": features["x"]},
                        name="construction_probability")


def run_train(cfg: Config) -> dict:
    df, n_unsure = _labelled_points(cfg)
    log.info("training on %d labels (%d construction, %d unsure skipped)",
             len(df), int(df["label"].sum()), n_unsure)
    before, after, lc_before, lc_after, changes, epsg = _analysis_layers(cfg)
    feats = build_features(before, after)
    rows, cols = df["row"].to_numpy(), df["col"].to_numpy()
    X, y = sample_matrix(feats, rows, cols), df["label"].to_numpy()
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        log.warning("dropping %d labelled points with missing data", int((~ok).sum()))
        df, X, y, rows, cols = df[ok].reset_index(drop=True), X[ok], y[ok], rows[ok], cols[ok]

    m, thr = cfg.model, cfg.model.probability_threshold
    make = _make_model(cfg)
    groups = spatial_blocks(df["x"], df["y"], m.block_size_m)
    p_spatial = out_of_fold_proba(make, X, y, groups, m.cv_folds, m.seed)
    p_random = out_of_fold_proba(make, X, y, None, m.cv_folds, m.seed)

    # Baselines need no training, so they are scored directly on every labelled point.
    ndvi_drop = feats["ndvi_diff"].values[rows, cols] < -cfg.change_threshold
    rule = changes.values[rows, cols] == 4
    methods = [
        {"method": f"Baseline: NDVI drop > {cfg.change_threshold}", "evaluation": "rule (no training)",
         **classification_metrics(y, ndvi_drop)},
        {"method": "Baseline: vegetation → built-up/bare rule", "evaluation": "rule (no training)",
         **classification_metrics(y, rule)},
        {"method": "Random forest", "evaluation": f"spatial CV ({m.block_size_m / 1000:g} km blocks)",
         **classification_metrics(y, p_spatial >= thr)},
        {"method": "Random forest", "evaluation": "random CV (optimistic, for comparison)",
         **classification_metrics(y, p_random >= thr)},
    ]

    log.info("fitting final model on all labels and predicting the whole area...")
    model = make().fit(X, y)
    proba = predict_map(model, feats)
    predicted = xr.where(proba >= thr, 1, 0).where(proba.notnull(), 0).astype("uint8").rename("construction")

    km2 = cfg.resolution**2 / 1e6
    strata = strata_map(lc_before, lc_after, changes).values
    strata_area = {name: float(np.sum(strata == code)) * km2 for code, name in STRATA.items()}
    area = {
        "stratified_estimate_from_labels": stratified_area_estimate(df[["stratum", "label"]], strata_area),
        "random_forest_map_km2": round(float(predicted.sum()) * km2, 2),
        "rule_veg_to_built_km2": round(float((changes == 4).sum()) * km2, 2),
        "ndvi_drop_km2": round(float((feats["ndvi_diff"] < -cfg.change_threshold).sum()) * km2, 2),
    }

    out = cfg.run_dir / "model"
    write_raster(proba, out / "construction_probability.tif", epsg)
    write_raster(predicted, out / "construction.tif", epsg)
    joblib.dump(model, out / "model.joblib")
    importance = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "new construction between the two periods (1) vs not (0), per 10 m pixel",
        "n_labels": len(df), "n_construction": int(y.sum()), "n_unsure_skipped": n_unsure,
        "labels_per_stratum": {name: {"labelled": int(len(g)), "construction": int(g.sum())}
                               for name, g in df.groupby("stratum")["label"]},
        "n_spatial_blocks": int(len(np.unique(groups))),
        "probability_threshold": thr,
        "methods": methods,
        "confusion_matrix_rf_spatial": confusion(y, p_spatial >= thr),
        "feature_importance": {k: round(float(v), 3) for k, v in importance},
        "area": area,
    }
    write_json(metrics, out / "metrics.json")
    catalog.upsert_item(cfg.catalog_dir, catalog.make_item(
        f"{cfg.aoi.name}-construction-model", cfg.aoi, None,
        assets={"probability": out / "construction_probability.tif", "construction": out / "construction.tif"},
        properties={"wch:fingerprint": cfg.input_fingerprint(), "wch:n_labels": len(df),
                    "wch:methods": methods},
    ))
    log.info("wrote model outputs to %s", out)
    return metrics