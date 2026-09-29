"""Config validation, GeoTIFF round-trip, and a headless smoke test of the Streamlit app."""
import json
from pathlib import Path

import numpy as np
import pytest
import rioxarray
import xarray as xr

from wch.change import ndvi_change
from wch.config import load_config
from wch.io import read_bands, write_bands, write_json, write_raster

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads():
    cfg = load_config(ROOT / "config" / "kanpur.yaml")
    assert cfg.aoi.name == "kanpur"
    assert cfg.before.stac_datetime == "2019-02-01/2019-03-31"
    assert cfg.run_dir == Path("data/processed/kanpur")


def test_config_rejects_huge_aoi(tmp_path):
    text = (ROOT / "config" / "kanpur.yaml").read_text().replace(
        "[80.22, 26.38, 80.46, 26.56]", "[79.0, 26.0, 81.0, 27.0]")
    p = tmp_path / "big.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="under"):
        load_config(p)


def test_geotiff_roundtrip_keeps_crs_and_location(tmp_path, scene_factory):
    # needs >= 2 pixels per axis: rioxarray infers pixel size from coordinate spacing
    ds = scene_factory(red=[[0.05, 0.2], [0.1, 0.1]], nir=[[0.4, 0.25], [0.3, 0.3]])
    ndvi = ndvi_change(ds, ds, 0.2, 100).after_ndvi.astype(np.float32)
    path = tmp_path / "ndvi.tif"
    write_raster(ndvi, path, epsg=32644)
    back = rioxarray.open_rasterio(path, masked=True)
    assert back.rio.crs.to_epsg() == 32644
    np.testing.assert_allclose(back.x.values, ds.x.values)
    np.testing.assert_allclose(back.y.values, ds.y.values)
    np.testing.assert_allclose(back.values[0], ndvi.values, rtol=1e-6)


def test_multiband_roundtrip_keeps_band_names(tmp_path, scene_factory):
    ds = scene_factory(red=[[0.05, np.nan], [0.1, 0.2]], nir=[[0.4, 0.3], [0.3, 0.3]])
    write_bands(ds, ("red", "nir"), tmp_path / "c.tif", epsg=32644)
    back = read_bands(tmp_path / "c.tif")
    assert list(back.data_vars) == ["red", "nir"]
    np.testing.assert_allclose(back["red"].values, ds["red"].values, equal_nan=True)


def _write_real_run(tmp_path, monkeypatch, with_timeseries=True):
    """Produce real pipeline outputs offline (fake downloads, see test_pipeline_offline)."""
    from dataclasses import replace
    import wch.pipeline as pipeline
    from test_pipeline_offline import fake_load, fake_selection, fake_timeseries_selection

    monkeypatch.setattr(pipeline, "select_scenes", lambda cfg: fake_selection())
    monkeypatch.setattr(pipeline, "select_timeseries_scenes", fake_timeseries_selection)
    monkeypatch.setattr(pipeline, "load_items", fake_load)
    cfg = replace(load_config(ROOT / "config" / "kanpur.yaml"),
                  output_dir=tmp_path / "processed", catalog_dir=tmp_path / "catalog")
    pipeline.run(cfg)
    if with_timeseries:
        pipeline.run_timeseries(cfg)
    return tmp_path / "processed"


@pytest.mark.parametrize("with_timeseries", [True, False])
def test_streamlit_app_renders(tmp_path, monkeypatch, with_timeseries):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    data_dir = _write_real_run(tmp_path, monkeypatch, with_timeseries)
    monkeypatch.setenv("WCH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("WCH_LABELS_DIR", str(tmp_path / "no_labels"))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "What Changed Here?"
    assert [t.label for t in at.tabs] == ["Map", "What changed", "Over time", "Label", "Model"]
    assert len(at.metric) >= 6          # change-type metrics + NDVI metrics
    info = " ".join(i.value for i in at.info)
    assert ("No time series yet" in info) != with_timeseries


def test_streamlit_app_empty_state(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WCH_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert "No results" in at.info[0].value