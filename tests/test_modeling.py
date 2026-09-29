"""Stage 4 tests: features, sampling, evaluation maths, label files, and a full
offline sample -> label -> train run on a synthetic landscape."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import wch.pipeline as pipeline
from wch.config import load_config
from wch.evaluate import (
    classification_metrics, out_of_fold_proba, spatial_blocks, stratified_area_estimate,
)
from wch.features import FEATURE_NAMES, build_features, local_mean, sample_matrix
from wch.landcover import BUILT, NODATA, SPARSE, VEG, WATER
from wch.modeling import (
    append_label, labels_path, points_path, read_labels, run_sample, run_train, undo_last_label,
)
from wch.sampling import allocate, sample_points, strata_map

ROOT = Path(__file__).resolve().parents[1]

# --- features -----------------------------------------------------------------------

def test_local_mean_matches_brute_force_and_ignores_nan():
    a = np.arange(1, 26, dtype="float32").reshape(5, 5)
    a[2, 2] = np.nan
    ref = np.array([[np.nanmean(a[max(0, i - 1):i + 2, max(0, j - 1):j + 2]) for j in range(5)]
                    for i in range(5)])
    np.testing.assert_allclose(local_mean(a, 3), ref, rtol=1e-6)


def test_build_features_names_and_values(scene_factory):
    before = scene_factory(red=[[0.05, 0.05]], nir=[[0.40, 0.40]])
    after = scene_factory(red=[[0.05, 0.20]], nir=[[0.40, 0.25]])
    for ds in (before, after):
        ds["swir16"] = ds["red"] * 0 + 0.2
    f = build_features(before, after)
    assert tuple(f.data_vars) == FEATURE_NAMES
    X = sample_matrix(f, np.array([0, 0]), np.array([0, 1]))
    i = FEATURE_NAMES.index("ndvi_diff")
    assert X[0, i] == pytest.approx(0.0) and X[1, i] < -0.5    # unchanged vs. vegetation lost

# --- sampling -------------------------------------------------------------------------

def test_strata_map_codes():
    b = xr.DataArray([[VEG, SPARSE, BUILT, VEG, SPARSE, WATER, NODATA]])
    a = xr.DataArray([[BUILT, BUILT, VEG, VEG, SPARSE, BUILT, VEG]])
    c = xr.DataArray([[4, 7, 5, 1, 7, 6, 0]])
    assert list(strata_map(b, a, c).values[0]) == [1, 2, 3, 4, 4, 3, 0]


def test_allocate_sums_exactly():
    counts = allocate(301, {"veg_to_built": 0.35, "sparse_to_built": 0.2, "other_change": 0.2, "stable": 0.25})
    assert sum(counts.values()) == 301
    assert counts == {"veg_to_built": 106, "sparse_to_built": 60, "other_change": 60, "stable": 75}  # leftover to largest remainder


def test_sample_points_respects_counts_and_spacing():
    strata = np.full((50, 50), 4, dtype="uint8")
    strata[:10, :10] = 1
    x, y = np.arange(50) * 10.0, np.arange(50)[::-1] * 10.0
    df = sample_points(strata, x, y, {"veg_to_built": 5, "stable": 10}, 30.0, np.random.default_rng(0))
    assert df["stratum"].value_counts().to_dict() == {"stable": 10, "veg_to_built": 5}
    pts = df[["x", "y"]].to_numpy()
    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    assert d[~np.eye(len(pts), dtype=bool)].min() >= 30.0
    assert (strata[df["row"], df["col"]] == np.where(df["stratum"] == "veg_to_built", 1, 4)).all()

# --- evaluation ------------------------------------------------------------------------

def test_spatial_blocks():
    b = spatial_blocks(np.array([0, 1999, 2000, 0]), np.array([0, 0, 0, 2500]), 2000)
    assert b[0] == b[1] and len({b[0], b[2], b[3]}) == 3


def test_spatial_cv_keeps_blocks_together():
    """Record which groups each fold trained on: a test block must never be in training."""
    X = np.random.default_rng(0).normal(size=(60, 3))
    y = np.tile([0, 1], 30)
    groups = np.repeat(np.arange(12), 5)
    seen = []

    class Recorder:
        def fit(self, Xtr, ytr):
            seen.append({tuple(r) for r in Xtr})
            return self

        def predict_proba(self, Xte):
            assert not any(tuple(r) in seen[-1] for r in Xte)
            return np.tile([0.5, 0.5], (len(Xte), 1))

    proba = out_of_fold_proba(Recorder, X, y, groups, n_splits=4, seed=0)
    assert not np.isnan(proba).any() and len(seen) == 4


def test_classification_metrics():
    m = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)


def test_stratified_area_estimate():
    labelled = pd.DataFrame({"stratum": ["a"] * 10 + ["b"] * 10,
                             "label": [1] * 5 + [0] * 5 + [1] + [0] * 9})
    est = stratified_area_estimate(labelled, {"a": 10.0, "b": 100.0, "empty": 0.0})
    assert est["estimate_km2"] == pytest.approx(10 * 0.5 + 100 * 0.1)
    lo, hi = est["ci95_km2"]
    assert lo < est["estimate_km2"] < hi
    assert stratified_area_estimate(labelled, {"a": 10.0, "c": 5.0})["estimate_km2"] is None

# --- label files -------------------------------------------------------------------------

def test_label_file_latest_answer_wins_and_undo(tmp_path):
    path = tmp_path / "l.csv"
    append_label(path, "p0001", 1)
    append_label(path, "p0002", 0)
    append_label(path, "p0001", 0)          # changed mind
    assert read_labels(path).set_index("id")["label"].to_dict() == {"p0002": 0, "p0001": 0}
    undo_last_label(path)
    assert read_labels(path).set_index("id")["label"].to_dict() == {"p0001": 1, "p0002": 0}

# --- full offline run: sample -> label -> train ---------------------------------------------

N = 60  # 60 x 60 pixels = 600 m
YS, XS = 2_927_000 - 10 * np.arange(N) - 5, 432_000 + 10 * np.arange(N) + 5
CONSTRUCTION = (slice(0, 30), slice(0, 30))   # bright concrete in 2026
FALLOW = (slice(30, 60), slice(0, 30))        # darker bare soil in 2026: the rule's false alarm


def fake_selection():
    from test_pipeline_offline import FakeItem, rows_for
    before = [FakeItem(f"b{i}", f"2019-02-{1 + 5 * i:02d}T05:30:00Z", "02.12") for i in range(3)]
    after = [FakeItem(f"a{i}", f"2026-02-{1 + 5 * i:02d}T05:30:00Z", "05.10") for i in range(3)]
    return {"before": (before, rows_for(before)), "after": (after, rows_for(after))}


def fake_load(items, bands, bbox, epsg, resolution, resampling=None):
    rng = np.random.default_rng(len(items))
    n = len(items)
    shape = (n, N, N)
    red, green, blue = np.full(shape, 0.05), np.full(shape, 0.06), np.full(shape, 0.04)
    nir, swir = np.full(shape, 0.40), np.full(shape, 0.15)
    offset = np.zeros(n)
    for t, item in enumerate(items):
        if item.id.startswith("a"):
            offset[t] = 1000
            for arr, v in ((red, 0.22), (green, 0.21), (blue, 0.20), (nir, 0.26), (swir, 0.32)):
                arr[t][CONSTRUCTION] = v
            for arr, v in ((red, 0.12), (green, 0.10), (blue, 0.08), (nir, 0.16), (swir, 0.28)):
                arr[t][FALLOW] = v
    coords = {"time": [np.datetime64(i.to_dict()["properties"]["datetime"][:19]) for i in items],
              "y": YS, "x": XS}

    def dn(r):
        noisy = r * (1 + rng.normal(0, 0.03, r.shape))
        return xr.DataArray((noisy * 10_000 + offset[:, None, None]).astype("uint16"),
                            dims=("time", "y", "x"), coords=coords)
    data = {"blue": dn(blue), "green": dn(green), "red": dn(red), "nir": dn(nir), "swir16": dn(swir),
            "scl": xr.DataArray(np.full(shape, 4, "uint8"), dims=("time", "y", "x"), coords=coords)}
    return xr.Dataset({b: data[b] for b in bands})


@pytest.fixture
def ml_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "select_scenes", lambda cfg: fake_selection())
    monkeypatch.setattr(pipeline, "load_items", fake_load)
    base = load_config(ROOT / "config" / "kanpur.yaml")
    return replace(
        base, output_dir=tmp_path / "processed", catalog_dir=tmp_path / "catalog",
        labels=replace(base.labels, labels_dir=tmp_path / "labels", n_points=80, min_spacing_m=25),
        model=replace(base.model, block_size_m=150, n_estimators=50),
    )


def label_from_truth(cfg):
    pts = pd.read_csv(points_path(cfg.labels.labels_dir, cfg.aoi.name))
    truth = np.zeros((N, N), dtype=int)
    truth[CONSTRUCTION] = 1
    for i, p in pts.iterrows():
        label = -1 if i == 0 else truth[p["row"], p["col"]]   # one "unsure" answer
        append_label(labels_path(cfg.labels.labels_dir, cfg.aoi.name), p["id"], label)
    return pts


def test_sample_label_train_offline(ml_cfg):
    pipeline.run(ml_cfg)
    result = run_sample(ml_cfg)
    assert result["points"] >= 40
    with pytest.raises(RuntimeError, match="already exists"):
        run_sample(ml_cfg)                           # protects existing points/labels
    label_from_truth(ml_cfg)

    m = run_train(ml_cfg)
    methods = {(r["method"], r["evaluation"].split(" ")[0]): r for r in m["methods"]}
    rf = methods[("Random forest", "spatial")]
    rule = methods[("Baseline: vegetation → built-up/bare rule", "rule")]
    assert m["n_unsure_skipped"] == 1
    assert rf["f1"] >= 0.9                           # learns concrete vs. fallow soil
    assert rule["recall"] == 1.0 and rule["precision"] < 0.8   # rule also flags the fallow field
    est = m["area"]["stratified_estimate_from_labels"]
    assert est["estimate_km2"] == pytest.approx(0.09, abs=0.03)    # true area: 900 px x 100 m2
    out = ml_cfg.run_dir / "model"
    assert all((out / f).exists() for f in ("construction_probability.tif", "construction.tif",
                                            "model.joblib", "metrics.json"))


def test_force_resample_backs_up_labels(ml_cfg):
    pipeline.run(ml_cfg)
    run_sample(ml_cfg)
    append_label(labels_path(ml_cfg.labels.labels_dir, ml_cfg.aoi.name), "p0001", 1)
    run_sample(ml_cfg, force=True)
    backups = list(Path(ml_cfg.labels.labels_dir).glob("*_backup_*.csv"))
    assert len(backups) == 1 and not labels_path(ml_cfg.labels.labels_dir, ml_cfg.aoi.name).exists()


def test_train_needs_labels(ml_cfg):
    pipeline.run(ml_cfg)
    run_sample(ml_cfg)
    with pytest.raises(RuntimeError, match="at least"):
        run_train(ml_cfg)


def test_app_label_and_model_tabs(ml_cfg, monkeypatch):
    """Click through the Label tab (answer, undo) and render the Model tab after training."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    pipeline.run(ml_cfg)
    run_sample(ml_cfg)
    monkeypatch.setenv("WCH_DATA_DIR", str(ml_cfg.output_dir))
    monkeypatch.setenv("WCH_LABELS_DIR", str(ml_cfg.labels.labels_dir))
    app = str(ROOT / "app" / "streamlit_app.py")
    lab_file = labels_path(ml_cfg.labels.labels_dir, ml_cfg.aoi.name)

    at = AppTest.from_file(app, default_timeout=90).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Map", "What changed", "Over time", "Label", "Model"]
    at.button(key="lab_yes").click().run()
    assert read_labels(lab_file)["label"].tolist() == [1]
    at.button(key="lab_no").click().run()
    assert read_labels(lab_file)["label"].tolist() == [1, 0]
    at.button(key="lab_undo").click().run()
    assert read_labels(lab_file)["label"].tolist() == [1]

    lab_file.unlink()
    label_from_truth(ml_cfg)
    run_train(ml_cfg)
    at = AppTest.from_file(app, default_timeout=90).run()
    assert not at.exception, at.exception
    assert any("All points labelled" in s.value for s in at.success)
    assert any("From your labels" == mt.label for mt in at.metric)