"""Offline unit tests for Stage 3: new indices, land-cover rules, change types,
time-series aggregation and anomaly flags."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wch.config import LandCover
from wch.display import colorize_classes
from wch.indices import mndwi, ndbi
from wch.landcover import BUILT, NODATA, SPARSE, VEG, WATER, change_type, classify, transition_matrix
from wch.search import select_monthly_dates
from wch.timeseries import flag_anomalies, monthly, season_summary, zone_series

RULES = LandCover(water_mndwi=0.0, vegetation_ndvi=0.4, bare_ndvi=0.2)


def scene(**bands):
    return xr.Dataset({k: xr.DataArray(np.asarray(v, dtype="float32"), dims=("y", "x"))
                       for k, v in bands.items()})

# --- indices ---------------------------------------------------------------------

def test_ndbi_and_mndwi_signs():
    ds = scene(green=[[0.06, 0.05, 0.10]], nir=[[0.02, 0.40, 0.25]], swir16=[[0.01, 0.20, 0.30]])
    assert ndbi(ds).values[0, 0] < 0 and mndwi(ds).values[0, 0] > 0     # water
    assert ndbi(ds).values[0, 1] < 0 and mndwi(ds).values[0, 1] < 0     # vegetation
    assert ndbi(ds).values[0, 2] > 0                                    # built-up/bare

# --- land cover ------------------------------------------------------------------

def test_classify_rules_and_nodata():
    # NDVI of the land pixels: 0.78 (veg), 0.11 (built), 0.30 (sparse)
    ds = scene(red=[[0.03, 0.05, 0.20, 0.07, np.nan]], nir=[[0.02, 0.40, 0.25, 0.13, 0.3]],
               green=[[0.06, 0.05, 0.20, 0.06, 0.1]], swir16=[[0.01, 0.20, 0.30, 0.15, 0.2]])
    assert list(classify(ds, RULES).values[0]) == [WATER, VEG, BUILT, SPARSE, NODATA]


def test_small_crop_difference_is_not_change():
    """Why the sparse class exists: NDVI 0.38 -> 0.42 must not count as land-use change."""
    def one(ndvi):  # nir/red giving the requested NDVI; no water
        red = 0.1
        return scene(red=[[red]], nir=[[red * (1 + ndvi) / (1 - ndvi)]], green=[[0.08]], swir16=[[0.2]])
    before, after = classify(one(0.38), RULES), classify(one(0.42), RULES)
    assert change_type(before, after).item() == 7                   # sparse -> not counted
    assert change_type(classify(one(0.6), RULES), classify(one(0.1), RULES)).item() == 4


def test_change_types():
    before = xr.DataArray([[VEG, BUILT, WATER, VEG, BUILT, WATER, VEG, NODATA, SPARSE, VEG, SPARSE]])
    after = xr.DataArray([[VEG, BUILT, WATER, BUILT, VEG, BUILT, WATER, VEG, BUILT, SPARSE, WATER]])
    assert list(change_type(before, after).values[0]) == [1, 2, 3, 4, 5, 6, 6, 0, 7, 7, 6]


def test_transition_matrix_areas():
    before = xr.DataArray([[VEG, VEG, BUILT]])
    after = xr.DataArray([[VEG, BUILT, BUILT]])
    m = transition_matrix(before, after, pixel_area_m2=1e6)   # 1 km2 pixels
    assert m.loc["vegetation", "built-up/bare"] == 1.0
    assert m.loc["vegetation", "vegetation"] == 1.0
    assert m.loc["built-up/bare", "built-up/bare"] == 1.0
    assert m.values.sum() == 3.0


def test_colorize_classes():
    rgba = colorize_classes(np.array([[0, 1]]), {1: "#ff0000"})
    assert list(rgba[0, 0]) == [0, 0, 0, 0] and list(rgba[0, 1]) == [255, 0, 0, 255]

# --- monthly date selection --------------------------------------------------------

def test_select_monthly_dates():
    def fake(id_, dt, cloud):
        return {"id": id_, "collection": "c1", "properties": {"datetime": dt, "eo:cloud_cover": cloud}}
    items = [fake("a", "2020-01-03T05:30:00Z", 5), fake("b", "2020-01-13T05:30:00Z", 1),
             fake("c", "2020-01-23T05:30:00Z", 2), fake("d", "2020-02-02T05:30:00Z", 9)]
    assert [r["date"] for r in select_monthly_dates(items, per_month=2)] == \
        ["2020-01-13", "2020-01-23", "2020-02-02"]

# --- time series -----------------------------------------------------------------

def test_zone_series_weights_and_clear_fraction():
    # 2 dates, 1x2 pixels. Zone weights: pixel 0 fully, pixel 1 half.
    stack = np.array([[[0.8, 0.2]], [[np.nan, 0.2]]])
    weights = {"z": np.array([[1.0, 0.5]])}
    df = zone_series(stack, ["2020-01-01", "2020-01-02"], weights, min_clear=0.5)
    d1, d2 = df.iloc[0], df.iloc[1]
    assert d1["ndvi"] == pytest.approx((0.8 * 1 + 0.2 * 0.5) / 1.5)
    assert d2["clear_fraction"] == pytest.approx(0.5 / 1.5, abs=1e-3)
    assert np.isnan(d2["ndvi"])                     # too little of the zone was clear


def _monthly_df(values_by_year, zone="stable vegetation", month=2):
    return pd.DataFrame({"zone": zone, "year": list(values_by_year),
                         "month": month, "ndvi": list(values_by_year.values()), "n_dates": 2})


def test_flag_anomalies_finds_outlier_year():
    df = flag_anomalies(_monthly_df({2019: 0.70, 2020: 0.72, 2021: 0.45, 2022: 0.71,
                                     2023: 0.69, 2024: 0.73}), z_threshold=3.5)
    assert df.loc[df["year"] == 2021, "is_anomaly"].item()
    assert df["is_anomaly"].sum() == 1


def test_flag_anomalies_needs_enough_years_and_stable_zone():
    few = flag_anomalies(_monthly_df({2019: 0.7, 2020: 0.2, 2021: 0.7}), 3.5)
    assert not few["is_anomaly"].any()               # 3 years is not enough to judge
    changed = flag_anomalies(_monthly_df({y: v for y, v in zip(range(2019, 2027), [0.7] * 4 + [0.1] * 4)},
                                         zone="vegetation -> built-up/bare"), 3.5)
    assert not changed["is_anomaly"].any() and changed["robust_z"].isna().all()


def test_monthly_and_season_summary():
    per_date = pd.DataFrame({"date": ["2020-02-05", "2020-02-20", "2020-03-05", "2021-02-10"],
                             "zone": "whole area", "ndvi": [0.6, 0.8, 0.5, np.nan]})
    m = monthly(per_date)
    assert m.loc[(m.year == 2020) & (m.month == 2), "ndvi"].item() == pytest.approx(0.7)
    assert len(m) == 2                                # the NaN-only month is dropped
    season = season_summary(m, [2, 3])
    assert season.loc["whole area", 2020] == pytest.approx(0.6)