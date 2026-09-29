"""Per-pixel features for the construction classifier (Stage 4).

Each pixel is described by 14 numbers computed from the two composites:

    ndvi / ndbi / mndwi / brightness   x   before, after, difference   (12)
    ndvi_diff_local, ndbi_diff_local   mean change in the 5x5 window    (2)

Why these: the indices summarise vegetation, built-up surfaces and water; the
differences capture change; brightness helps because fresh concrete and cleared
sites are bright. The two "local" features give context: real construction
comes in patches, while noise (a single odd pixel) does not.
All values are plain floats, so a tree model needs no scaling.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from wch.indices import mndwi, ndbi, ndvi

LOCAL_WINDOW = 5  # pixels (5 x 10 m = 50 m)

FEATURE_NAMES = (
    "ndvi_before", "ndvi_after", "ndvi_diff",
    "ndbi_before", "ndbi_after", "ndbi_diff",
    "mndwi_before", "mndwi_after", "mndwi_diff",
    "bright_before", "bright_after", "bright_diff",
    "ndvi_diff_local", "ndbi_diff_local",
)


def local_mean(arr: np.ndarray, size: int = LOCAL_WINDOW) -> np.ndarray:
    """Mean of the size x size window around each pixel, ignoring NaN.

    Uses an 'integral image' (2-D cumulative sum): any window sum is then four
    lookups, so this is fast and needs no extra libraries.
    """
    r = size // 2
    valid = ~np.isnan(arr)
    h, w = arr.shape

    def box_sum(a: np.ndarray) -> np.ndarray:
        c = np.pad(a, r).cumsum(0).cumsum(1)
        c = np.pad(c, ((1, 0), (1, 0)))
        return (c[size:size + h, size:size + w] - c[0:h, size:size + w]
                - c[size:size + h, 0:w] + c[0:h, 0:w])

    total = box_sum(np.where(valid, arr, 0.0).astype("float64"))
    count = box_sum(valid.astype("float64"))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan).astype("float32")


def _brightness(ds: xr.Dataset) -> xr.DataArray:
    return (ds["red"] + ds["green"] + ds["blue"]) / 3


def build_features(before: xr.Dataset, after: xr.Dataset) -> xr.Dataset:
    """Two composites (same grid) -> Dataset with one float32 variable per feature."""
    before, after = xr.align(before, after, join="exact")
    out = {}
    for name, fn in (("ndvi", ndvi), ("ndbi", ndbi), ("mndwi", mndwi), ("bright", _brightness)):
        b, a = fn(before), fn(after)
        out[f"{name}_before"], out[f"{name}_after"], out[f"{name}_diff"] = b, a, a - b
    for name in ("ndvi_diff", "ndbi_diff"):
        out[f"{name}_local"] = xr.DataArray(
            local_mean(np.asarray(out[name].values, dtype="float32")),
            dims=("y", "x"), coords={"y": before["y"], "x": before["x"]},
        )
    return xr.Dataset({k: out[k].astype("float32") for k in FEATURE_NAMES})


def sample_matrix(features: xr.Dataset, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Feature values at given pixel positions -> array (n_points, n_features)."""
    return np.column_stack([features[n].values[rows, cols] for n in FEATURE_NAMES])


def iter_row_blocks(features: xr.Dataset, block_rows: int = 256):
    """Yield (row_start, row_stop, X) over the whole grid, for memory-friendly prediction."""
    n_rows = features.sizes["y"]
    for r0 in range(0, n_rows, block_rows):
        r1 = min(r0 + block_rows, n_rows)
        X = np.column_stack([features[n].values[r0:r1].ravel() for n in FEATURE_NAMES])
        yield r0, r1, X