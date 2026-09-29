"""Rule-based land cover per period, and what changed into what.

Four classes from two indices (thresholds in the config):
    water              MNDWI > water_mndwi
    vegetation         NDVI >= vegetation_ndvi   (and not water)
    built-up/bare      NDVI <  bare_ndvi         (and not water)
    sparse vegetation  in between

Why a middle class: with a single NDVI cut-off, a field at 0.38 one year and 0.42
the next would flip between "built-up" and "vegetation", turning ordinary crop
differences into fake land-use change. With a buffer, "vegetation -> built-up"
needs a drop from >= 0.4 to < 0.2, which is a large, robust signal. Pixels that
are sparse in either year are reported separately and never counted as change.

"Built-up/bare" honestly mixes buildings, roads, bare soil and river sand: in a
Feb-Mar composite almost all farmland is under crop, so non-vegetated land is
mostly built-up, but fallow fields and sandbars also land here. Stage 4 trains a
classifier to separate real construction from these look-alikes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from wch.config import LandCover
from wch.indices import mndwi, ndvi

NODATA, WATER, BUILT, VEG, SPARSE = 0, 1, 2, 3, 4
# Dict order = row/column order in the transition table (least to most vegetated).
CLASS_NAMES = {WATER: "water", BUILT: "built-up/bare", SPARSE: "sparse vegetation", VEG: "vegetation"}

# Change types: what happened to each pixel between the two periods.
CHANGE_NAMES = {
    1: "stable vegetation",
    2: "stable built-up/bare",
    3: "stable water",
    4: "vegetation -> built-up/bare",   # candidate urban expansion
    5: "built-up/bare -> vegetation",
    6: "water change",                  # to or from water (river shifts, ponds)
    7: "sparse vegetation (either year)",  # ambiguous middle range: not counted as change
}


def classify(ds: xr.Dataset, rules: LandCover) -> xr.DataArray:
    """Composite (reflectance bands) -> uint8 class map (0 = no data)."""
    v, w = ndvi(ds), mndwi(ds)
    land = xr.where(v >= rules.vegetation_ndvi, VEG, xr.where(v < rules.bare_ndvi, BUILT, SPARSE))
    cls = xr.where(w > rules.water_mndwi, WATER, land)
    valid = v.notnull() & w.notnull()
    return cls.where(valid, NODATA).astype("uint8").rename("landcover")


def change_type(before: xr.DataArray, after: xr.DataArray) -> xr.DataArray:
    """Two class maps -> uint8 change-type map (codes in CHANGE_NAMES, 0 = no data)."""
    b, a = before, after
    out = xr.full_like(b, 0, dtype="uint8")
    out = xr.where((b == VEG) & (a == VEG), 1, out)
    out = xr.where((b == BUILT) & (a == BUILT), 2, out)
    out = xr.where((b == WATER) & (a == WATER), 3, out)
    out = xr.where((b == VEG) & (a == BUILT), 4, out)
    out = xr.where((b == BUILT) & (a == VEG), 5, out)
    out = xr.where((b == SPARSE) | (a == SPARSE), 7, out)
    out = xr.where((b != a) & ((b == WATER) | (a == WATER)), 6, out)   # water change wins
    out = xr.where((b == NODATA) | (a == NODATA), 0, out)
    return out.astype("uint8").rename("change_type")


def transition_matrix(before: xr.DataArray, after: xr.DataArray, pixel_area_m2: float) -> pd.DataFrame:
    """Area (km2) for every before-class (rows) -> after-class (columns) pair."""
    b, a = np.asarray(before).ravel(), np.asarray(after).ravel()
    names = list(CLASS_NAMES.values())
    m = pd.DataFrame(0.0, index=names, columns=names)
    for i, ni in CLASS_NAMES.items():
        for j, nj in CLASS_NAMES.items():
            m.loc[ni, nj] = round(np.sum((b == i) & (a == j)) * pixel_area_m2 / 1e6, 2)
    return m


def landcover_stats(
    before: xr.DataArray, after: xr.DataArray, changes: xr.DataArray,
    ndvi_diff: xr.DataArray, pixel_area_m2: float,
) -> dict:
    km2 = pixel_area_m2 / 1e6
    c = np.asarray(changes)
    valid = c > 0
    n_valid = int(valid.sum())

    def area(arr, code):
        return round(float(np.sum(np.asarray(arr) == code)) * km2, 2)

    change_areas = {
        name: {"km2": round(float(np.sum(c == code)) * km2, 2),
               "pct": round(100 * float(np.sum(c == code)) / max(n_valid, 1), 2)}
        for code, name in CHANGE_NAMES.items()
    }
    # Reference check: land that is built-up/bare in BOTH years should not get greener.
    # A consistent NDVI shift here points to processing/sensor differences between years.
    stable_built = np.asarray(ndvi_diff)[c == 2]
    stable_built = stable_built[~np.isnan(stable_built)]
    stable_veg = np.asarray(ndvi_diff)[c == 1]
    stable_veg = stable_veg[~np.isnan(stable_veg)]
    return {
        "class_area_km2": {
            "before": {n: area(before, k) for k, n in CLASS_NAMES.items()},
            "after": {n: area(after, k) for k, n in CLASS_NAMES.items()},
        },
        "change_types": change_areas,
        "transitions_km2": transition_matrix(before, after, pixel_area_m2).to_dict(orient="index"),
        "median_ndvi_diff_stable_built": round(float(np.median(stable_built)), 4) if stable_built.size else None,
        "median_ndvi_diff_stable_vegetation": round(float(np.median(stable_veg)), 4) if stable_veg.size else None,
    }