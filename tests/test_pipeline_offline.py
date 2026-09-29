"""End-to-end pipeline tests without network: search and pixel loading are replaced
by fakes, everything else (radiometry, masking, compositing, change, land cover,
time series, files, catalog, caching) runs for real."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray
import xarray as xr

import wch.pipeline as pipeline
from wch.config import load_config

ROOT = Path(__file__).resolve().parents[1]
Y = 2_927_000 - 10 * np.arange(4) - 5   # 4x4 pixels near Kanpur (UTM 44N)
X = 432_000 + 10 * np.arange(4) + 5


class FakeItem:
    def __init__(self, id_, dt, baseline, nir_field=0.40):
        self.id = id_
        self.nir_field = nir_field   # lets time-series tests vary crop greenness per date
        self._d = {"id": id_, "collection": "fake", "assets": {},
                   "properties": {"datetime": dt, "eo:cloud_cover": 1.0,
                                  "s2:processing_baseline": baseline}}

    def to_dict(self):
        return self._d


def rows_for(items):
    return [{"date": i.to_dict()["properties"]["datetime"][:10], "item_ids": [i.id],
             "mean_cloud_cover": 1.0, "collection": "fake"} for i in items]


def fake_selection():
    """2019: 3 dates, old baseline (no offset). 2026: 3 dates, new baseline (-1000 DN)."""
    before = [FakeItem(f"b{i}", f"2019-02-{1 + 5 * i:02d}T05:30:00Z", "02.12") for i in range(3)]
    after = [FakeItem(f"a{i}", f"2026-02-{1 + 5 * i:02d}T05:30:00Z", "05.10") for i in range(3)]
    return {"before": (before, rows_for(before)), "after": (after, rows_for(after))}


def fake_load(items, bands, bbox, epsg, resolution, resampling=None):
    """4x4 pixels. Left half: field that becomes built-up from 2023 on.
    Pixel (3, 3): water in every year. One early date has a cloud over pixel (0, 0)."""
    n = len(items)
    red, green = np.full((n, 4, 4), 0.05), np.full((n, 4, 4), 0.05)
    nir, swir = np.zeros((n, 4, 4)), np.full((n, 4, 4), 0.20)
    scl = np.full((n, 4, 4), 4, dtype="uint8")
    offset_dn = np.zeros(n)
    for t, item in enumerate(items):
        year = int(item.to_dict()["properties"]["datetime"][:4])
        nir[t] = item.nir_field
        if year >= 2023:
            red[t, :, :2], nir[t, :, :2], green[t, :, :2], swir[t, :, :2] = 0.20, 0.25, 0.20, 0.30
        if float(item.to_dict()["properties"]["s2:processing_baseline"]) >= 4:
            offset_dn[t] = 1000
    red[:, 3, 3], nir[:, 3, 3], green[:, 3, 3], swir[:, 3, 3] = 0.03, 0.02, 0.06, 0.01  # water
    if n > 1 and items[1].id == "b1":
        red[1, 0, 0], nir[1, 0, 0], scl[1, 0, 0] = 0.6, 0.6, 9                          # cloud
    coords = {"time": [np.datetime64(i.to_dict()["properties"]["datetime"][:19]) for i in items],
              "y": Y, "x": X}

    def dn(r):
        return xr.DataArray((r * 10_000 + offset_dn[:, None, None]).astype("uint16"),
                            dims=("time", "y", "x"), coords=coords)
    data = {"blue": dn(red), "green": dn(green), "red": dn(red), "nir": dn(nir), "swir16": dn(swir),
            "scl": xr.DataArray(scl, dims=("time", "y", "x"), coords=coords)}
    return xr.Dataset({b: data[b] for b in bands})


def fake_timeseries_selection(cfg):
    """Feb and Mar of 2019-2026; Feb 2021 is a drought month (fields much less green)."""
    items = [FakeItem(f"t{y}{m}", f"{y}-{m:02d}-10T05:30:00Z", "05.10",
                      nir_field=0.15 if (y, m) == (2021, 2) else 0.40)
             for y in range(2019, 2027) for m in (2, 3)]
    return items, rows_for(items)


@pytest.fixture
def cfg(tmp_path):
    base = load_config(ROOT / "config" / "kanpur.yaml")
    return replace(base, output_dir=tmp_path / "processed", catalog_dir=tmp_path / "catalog")


@pytest.fixture
def fakes(monkeypatch):
    calls = {"load": 0}

    def counting_load(*args, **kwargs):
        calls["load"] += 1
        return fake_load(*args, **kwargs)

    monkeypatch.setattr(pipeline, "select_scenes", lambda cfg: fake_selection())
    monkeypatch.setattr(pipeline, "select_timeseries_scenes", fake_timeseries_selection)
    monkeypatch.setattr(pipeline, "load_items", counting_load)
    return calls


def read(path):
    return rioxarray.open_rasterio(path).values[0]


def test_full_run_offline(cfg, fakes):
    summary = pipeline.run(cfg)
    s = summary["stats"]
    assert s["valid_pixels"] == 16
    assert s["vegetation_loss_pct"] == pytest.approx(50.0)   # left half became built-up
    assert s["vegetation_gain_pct"] == 0.0
    # Offset handled: right half is unchanged despite the 2026 +1000 DN shift.
    diff = read(cfg.run_dir / "ndvi_diff.tif")
    np.testing.assert_allclose(diff[:, 2:], 0.0, atol=1e-5)
    # Cloud handled: pixel (0, 0) keeps its normal 2019 value.
    assert read(cfg.run_dir / "before_ndvi.tif")[0, 0] == pytest.approx((0.40 - 0.05) / 0.45, abs=1e-4)

    # Land cover: left half vegetation -> built-up/bare (4), pixel (3,3) stable water (3)
    changes = read(cfg.run_dir / "change_type.tif")
    assert (changes[:, :2] == 4).all()
    assert changes[3, 3] == 3
    assert (changes[:3, 2:] == 1).all()
    assert summary["landcover"]["median_ndvi_diff_stable_vegetation"] == pytest.approx(0.0, abs=1e-4)

    p = summary["periods"]
    assert p["before"]["mean_clear_obs"] == pytest.approx((15 * 3 + 2) / 16, abs=0.01)
    assert set(p["after"]["offset_rules"].values()) == {"baseline 5.10 >= 04.00 rule"}
    assert (cfg.catalog_dir / "catalog.json").exists()
    assert json.loads((cfg.run_dir / "summary.json").read_text())["fingerprint"] == cfg.input_fingerprint()
    assert fakes["load"] == 2


def test_second_run_uses_cache_and_force_rebuilds(cfg, fakes):
    pipeline.run(cfg)
    pipeline.run(cfg)
    assert fakes["load"] == 2          # nothing re-downloaded
    pipeline.run(cfg, force=True)
    assert fakes["load"] == 4


def test_changed_setting_rebuilds(cfg, fakes):
    pipeline.run(cfg)
    changed = replace(cfg, compositing=replace(cfg.compositing, max_scenes=5))
    pipeline.run(changed)
    assert fakes["load"] == 4


def test_timeseries_requires_run_first(cfg, fakes):
    with pytest.raises(RuntimeError, match="wch run"):
        pipeline.run_timeseries(cfg)


def test_timeseries_offline(cfg, fakes):
    pipeline.run(cfg)
    result = pipeline.run_timeseries(cfg)
    assert result["n_dates"] == 16

    monthly = pd.read_csv(cfg.run_dir / "timeseries" / "monthly.csv")
    new_built = monthly[(monthly["zone"] == "vegetation -> built-up/bare") & (monthly["month"] == 3)]
    before_2023 = new_built[new_built["year"] < 2023]["ndvi"]
    after_2023 = new_built[new_built["year"] >= 2023]["ndvi"]
    assert before_2023.min() > 0.7 and after_2023.max() < 0.2   # the series shows WHEN it changed

    # The drought month is flagged for stable vegetation, and nothing else is.
    flagged = {(a["zone"], a["year"], a["month"]) for a in result["anomalies"]}
    assert ("stable vegetation", 2021, 2) in flagged
    assert all((y, m) == (2021, 2) for _, y, m in flagged)

    pipeline.run_timeseries(cfg)          # second call reuses the cached stack
    assert fakes["load"] == 3             # 2 composites + 1 time-series download