"""Offline unit tests for geo, radiometry, indices, change, search selection and display."""
import numpy as np
import pytest
import xarray as xr

from wch.change import ndvi_change
from wch.display import colorize, stretch_rgb
from wch.geo import bbox_size_km, utm_epsg
from wch.indices import normalized_difference
from wch.radiometry import Radiometry, radiometry_for, radiometry_for_items, to_reflectance
from wch.search import day_of_year_gap, pick_matched_dates, prefer_collections, rank_dates

# --- geo -------------------------------------------------------------------

@pytest.mark.parametrize("lon, lat, expected", [
    (80.33, 26.45, 32644),    # Kanpur -> UTM 44N
    (-0.12, 51.5, 32630),     # London -> 30N
    (151.2, -33.9, 32756),    # Sydney -> 56S
])
def test_utm_epsg(lon, lat, expected):
    assert utm_epsg(lon, lat) == expected


def test_bbox_size_km_kanpur():
    w, h = bbox_size_km((80.22, 26.38, 80.46, 26.56))
    assert 23 < w < 25 and 19 < h < 21

# --- radiometry --------------------------------------------------------------

def item(props=None, band_meta=None, key="red"):
    asset = {} if band_meta is None else {"raster:bands": [band_meta]}
    return {"properties": props or {}, "assets": {key: asset}}


def test_provider_applied_offset_wins():
    it = item({"earthsearch:boa_offset_applied": True, "s2:processing_baseline": "05.10"})
    assert radiometry_for(it, "red").offset == 0.0


def test_band_metadata_used():
    it = item({"s2:processing_baseline": "05.10"}, {"scale": 0.0001, "offset": -0.1})
    rad = radiometry_for(it, "red")
    assert (rad.scale, rad.offset, rad.source) == (0.0001, -0.1, "STAC band metadata")


def test_stac_1_1_bands_field():
    it = {"properties": {}, "assets": {"red": {"bands": [{"scale": 0.0001, "offset": -0.1}]}}}
    assert radiometry_for(it, "red").offset == -0.1


@pytest.mark.parametrize("baseline, expected", [("05.10", -0.1), ("04.00", -0.1), ("02.12", 0.0)])
def test_baseline_fallback(baseline, expected):
    assert radiometry_for(item({"s2:processing_baseline": baseline}), "red").offset == expected


def test_items_must_agree():
    old = item({"s2:processing_baseline": "02.12"})
    new = item({"s2:processing_baseline": "05.10"})
    with pytest.raises(ValueError):
        radiometry_for_items([old, new], "red")


def test_to_reflectance_offset_and_nodata():
    dn = xr.DataArray(np.array([[0, 2000, 500]], dtype="uint16"), dims=("y", "x"))
    refl = to_reflectance(dn, Radiometry(1e-4, -0.1, "test")).values
    assert np.isnan(refl[0, 0])                   # DN 0 = no data
    assert refl[0, 1] == pytest.approx(0.1)       # 2000 * 1e-4 - 0.1
    assert refl[0, 2] == 0.0                      # negative clipped to 0


def test_offset_matters_for_ndvi():
    """Why we care: the same surface gives different NDVI if the offset is ignored."""
    red_dn, nir_dn = 1000 + 500, 1000 + 3000     # post-2022 DNs for red=0.05, nir=0.30
    true = (0.30 - 0.05) / (0.30 + 0.05)
    naive = (nir_dn - red_dn) / (nir_dn + red_dn)
    assert abs(true - naive) > 0.1

# --- indices & change --------------------------------------------------------

def test_normalized_difference_values_and_zero_denominator():
    a = xr.DataArray([[0.3, 0.0, np.nan]], dims=("y", "x"))
    b = xr.DataArray([[0.1, 0.0, 0.1]], dims=("y", "x"))
    nd = normalized_difference(a, b).values
    assert nd[0, 0] == pytest.approx(0.5)
    assert np.isnan(nd[0, 1]) and np.isnan(nd[0, 2])


def test_ndvi_change_stats(scene_factory):
    # 4 pixels: stays green, green->built (loss), bare->green (gain), no data after
    before = scene_factory(red=[[0.05, 0.05, 0.20, 0.05]], nir=[[0.40, 0.40, 0.25, 0.40]])
    after = scene_factory(red=[[0.05, 0.20, 0.05, np.nan]], nir=[[0.40, 0.25, 0.40, np.nan]])
    res = ndvi_change(before, after, threshold=0.2, pixel_area_m2=1e6)  # 1 km2 pixels
    s = res.stats
    assert s["valid_pixels"] == 3
    assert s["vegetation_loss_pct"] == pytest.approx(100 / 3, abs=0.01)
    assert s["vegetation_gain_pct"] == pytest.approx(100 / 3, abs=0.01)
    assert s["vegetation_loss_km2"] == pytest.approx(1.0)
    assert s["valid_area_km2"] == pytest.approx(3.0)
    assert np.isnan(res.ndvi_diff.values[0, 3])


def test_misaligned_grids_raise(scene_factory):
    a = scene_factory(red=[[0.1, 0.1]], nir=[[0.3, 0.3]])
    b = a.assign_coords(x=a.x + 10)  # shifted by one pixel
    with pytest.raises(ValueError):
        ndvi_change(a, b, 0.2, 100)

# --- scene selection -----------------------------------------------------------

def fake(id_, dt, cloud, collection="c1"):
    return {"id": id_, "collection": collection, "properties": {"datetime": dt, "eo:cloud_cover": cloud}}


def test_rank_dates_groups_tiles_by_day():
    items = [
        fake("a1", "2019-02-10T05:30:00Z", 5.0), fake("a2", "2019-02-10T05:30:05Z", 15.0),  # mean 10
        fake("b1", "2019-02-15T05:30:00Z", 8.0),
        fake("c1", "2019-03-01T05:30:00Z", 30.0),
    ]
    rows = rank_dates(items)
    assert [r["date"] for r in rows] == ["2019-02-15", "2019-02-10", "2019-03-01"]
    assert rows[1]["item_ids"] == ["a1", "a2"]


def test_prefer_collections_one_collection_per_date():
    items = [
        fake("new", "2019-03-23T05:30:00Z", 0, "c1"), fake("old", "2019-03-23T05:30:00Z", 0, "old"),
        fake("only_old", "2019-02-11T05:30:00Z", 0, "old"),
    ]
    kept = {i["id"] for i in prefer_collections(items, ("c1", "old"))}
    assert kept == {"new", "only_old"}


def test_day_of_year_gap_ignores_year_and_wraps():
    assert day_of_year_gap("2019-02-11", "2026-02-09") == 2
    assert day_of_year_gap("2019-12-30", "2026-01-02") == 3


def test_pick_matched_dates_prefers_same_season_over_order():
    """The real Kanpur case: without matching we'd compare late March with early Feb."""
    before = [fake("b_mar", "2019-03-23T05:30:00Z", 0.0), fake("b_feb", "2019-02-11T05:30:00Z", 1.0)]
    after = [fake("a_feb", "2026-02-06T05:30:00Z", 0.0), fake("a_feb2", "2026-02-11T05:30:00Z", 0.5),
             fake("a_mar", "2026-03-25T05:30:00Z", 3.0)]
    b, a, gap = pick_matched_dates(before, after, cloud_limit=5)
    assert (b["date"], a["date"], gap) == ("2019-02-11", "2026-02-11", 0)


def test_pick_matched_dates_respects_cloud_limit():
    before = [fake("b_clear", "2019-02-01T05:30:00Z", 0.0), fake("b_cloudy", "2019-02-20T05:30:00Z", 15.0)]
    after = [fake("a", "2026-02-20T05:30:00Z", 0.0)]
    b, _, gap = pick_matched_dates(before, after, cloud_limit=5)
    assert b["date"] == "2019-02-01" and gap == 19


# --- display -------------------------------------------------------------------

def test_stretch_rgb_reserves_zero_for_nodata():
    r = np.array([[np.nan, 0.0, 0.3, 1.0]])
    rgb = stretch_rgb(r, r, r)
    assert rgb.shape == (3, 1, 4)
    assert list(rgb[0, 0]) == [0, 1, 255, 255]


def test_colorize_nan_transparent():
    rgba = colorize(np.array([[np.nan, 0.0]]), "RdBu", -0.5, 0.5)
    assert rgba.shape == (1, 2, 4)
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 255