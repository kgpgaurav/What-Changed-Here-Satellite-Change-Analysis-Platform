"""Offline tests for Stage 2: cloud masking, compositing, date selection, catalog, config."""
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wch import catalog
from wch.composite import clear_mask, median_composite, reflectance_stack
from wch.config import load_config
from wch.radiometry import Radiometry
from wch.search import median_day_of_year, select_composite_dates

ROOT = Path(__file__).resolve().parents[1]


def stack(values, name="red"):
    """(time, y=1, x=n) DataArray from a list of per-date rows."""
    arr = np.asarray(values, dtype="float32")[:, None, :]
    return xr.DataArray(arr, dims=("time", "y", "x"),
                        coords={"time": np.arange(arr.shape[0]), "y": [0], "x": np.arange(arr.shape[2])},
                        name=name)

# --- cloud mask ------------------------------------------------------------------

def test_clear_mask_classes():
    scl = xr.DataArray(np.arange(12, dtype="uint8"))
    clear = clear_mask(scl).values
    assert [i for i, ok in enumerate(clear) if ok] == [4, 5, 6, 7, 11]

# --- reflectance per date ----------------------------------------------------------

def test_reflectance_stack_uses_each_dates_offset():
    raw = xr.Dataset({"red": stack([[500], [1500]]).astype("uint16")})
    rads = {"red": [Radiometry(1e-4, 0.0, "old"), Radiometry(1e-4, -0.1, "new")]}
    refl = reflectance_stack(raw, ("red",), rads)["red"].values.ravel()
    np.testing.assert_allclose(refl, [0.05, 0.05], atol=1e-6)  # same surface, same reflectance

# --- median composite --------------------------------------------------------------

def test_median_composite_ignores_clouds_and_counts_obs():
    # 3 dates x 3 pixels. Pixel 0: 3 clear obs. Pixel 1: cloud on date 2 (bright 0.9).
    # Pixel 2: cloudy on 2 of 3 dates -> only 1 clear obs.
    red = stack([[0.10, 0.10, 0.10], [0.12, 0.90, 0.50], [0.14, 0.12, 0.60]])
    refl = xr.Dataset({"red": red, "nir": red * 3})
    scl = stack([[4, 4, 9], [4, 9, 9], [4, 4, 4]]).astype("uint8")
    comp, n = median_composite(refl, clear_mask(scl), min_clear_obs=2)
    assert list(n.values.ravel()) == [3, 2, 1]
    np.testing.assert_allclose(comp["red"].values.ravel()[:2], [0.12, 0.11], atol=1e-6)
    assert np.isnan(comp["red"].values.ravel()[2])  # 1 obs < min_clear_obs


def test_median_resists_missed_cloud():
    """A cloud the mask missed (bright outlier) barely affects the median."""
    red = stack([[0.10], [0.11], [0.95], [0.12], [0.10]])
    comp, _ = median_composite(xr.Dataset({"red": red}), xr.ones_like(red, dtype=bool), 1)
    assert comp["red"].item() == pytest.approx(0.11)

# --- date selection ----------------------------------------------------------------

def fake(id_, dt, cloud):
    return {"id": id_, "collection": "c1", "properties": {"datetime": dt, "eo:cloud_cover": cloud}}


def test_select_composite_dates_one_per_slice_then_fill():
    items = [fake("a", "2019-02-01T05:30:00Z", 10), fake("b", "2019-02-06T05:30:00Z", 1),
             fake("c", "2019-02-11T05:30:00Z", 15), fake("d", "2019-03-01T05:30:00Z", 0)]
    rows = select_composite_dates(items, 3, date(2019, 2, 1), date(2019, 3, 31))
    # slice 1 (Feb 1-19): clearest is b; slice 2 (Feb 20-Mar 11): d; slice 3 empty -> fill with a
    assert [r["date"] for r in rows] == ["2019-02-01", "2019-02-06", "2019-03-01"]


def test_select_composite_dates_spreads_across_season():
    """The real Kanpur 2026 case: 20 dates, all 0% cloud. Picking 'clearest first'
    gave 8 February dates; spreading must cover both months."""
    start = date(2026, 2, 1)
    items = [fake(f"i{k}", f"{start + timedelta(days=3 * k)}T05:30:00Z", 0.0) for k in range(20)]
    rows = select_composite_dates(items, 8, start, date(2026, 3, 31))
    dates = [r["date"] for r in rows]
    assert len(dates) == 8
    assert dates[0] < "2026-02-10" and dates[-1] > "2026-03-20"
    assert 50 <= median_day_of_year(dates) <= 70  # centred mid-window (~day 60)


def test_median_day_of_year():
    assert median_day_of_year(["2019-02-01", "2019-02-11", "2019-03-31"]) == 42

# --- config ------------------------------------------------------------------------

def test_compositing_config_and_fingerprint():
    cfg = load_config(ROOT / "config" / "kanpur.yaml")
    assert cfg.compositing.method == "median" and cfg.compositing.max_scenes == 8
    fp = cfg.input_fingerprint()
    # Changing a later-stage setting keeps the cache valid...
    assert replace(cfg, change_threshold=0.3).input_fingerprint() == fp
    # ...changing an input setting invalidates it.
    new_before = replace(cfg.before, start=date(2019, 1, 15))
    assert replace(cfg, before=new_before).input_fingerprint() != fp
    single = replace(cfg, compositing=replace(cfg.compositing, method="single"))
    assert single.compositing.effective_min_clear_obs == 1


def test_code_version_invalidates_cache(monkeypatch):
    """A code change that alters results (new PIPELINE_VERSION) must change the fingerprint."""
    import wch.config as config_module
    cfg = load_config(ROOT / "config" / "kanpur.yaml")
    fp = cfg.input_fingerprint()
    monkeypatch.setattr(config_module, "PIPELINE_VERSION", "999")
    assert cfg.input_fingerprint() != fp


# --- catalog -------------------------------------------------------------------------

def test_catalog_upsert_find_and_relative_hrefs(tmp_path):
    cfg = load_config(ROOT / "config" / "kanpur.yaml")
    data = tmp_path / "processed" / "x.tif"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"fake")
    root = tmp_path / "catalog"

    for version in ("v1", "v2"):  # second write replaces the first
        item = catalog.make_item("kanpur-before-composite", cfg.aoi, cfg.before,
                                 {"reflectance": data}, {"wch:fingerprint": version})
        catalog.upsert_item(root, item)

    found = catalog.find_item(root, "kanpur-before-composite")
    assert found.properties["wch:fingerprint"] == "v2"
    assert len(list(catalog.open_catalog(root).get_items())) == 1
    assert catalog.asset_path(found, "reflectance").resolve() == data.resolve()
    saved = (root / "kanpur-before-composite" / "kanpur-before-composite.json").read_text()
    assert str(tmp_path) not in saved  # hrefs are relative, so data/ can be moved
    assert catalog.find_item(root, "missing") is None