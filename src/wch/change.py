"""Compare two periods: NDVI difference map and summary statistics.

Pure computation on in-memory arrays (no network, no files), so it is easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from wch.indices import ndvi


@dataclass
class ChangeResult:
    before_ndvi: xr.DataArray
    after_ndvi: xr.DataArray
    ndvi_diff: xr.DataArray
    stats: dict


def change_stats(
    before_ndvi: xr.DataArray,
    after_ndvi: xr.DataArray,
    diff: xr.DataArray,
    threshold: float,
    pixel_area_m2: float,
) -> dict:
    """Areas are over pixels valid in BOTH periods; percentages are of that valid area.

    'Loss'/'gain' mean a large NDVI drop/rise. A drop is consistent with
    construction but can also be a fallow field or river change; the label
    is intentionally about vegetation, not about buildings.
    """
    valid = diff.notnull()
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("no pixels are valid in both periods")
    km2 = pixel_area_m2 / 1e6
    n_loss = int((diff < -threshold).sum())  # NaN comparisons are False, so NaNs are excluded
    n_gain = int((diff > threshold).sum())
    return {
        "threshold": threshold,
        "valid_pixels": n_valid,
        "valid_area_km2": round(n_valid * km2, 2),
        "mean_ndvi_before": round(float(before_ndvi.where(valid).mean()), 4),
        "mean_ndvi_after": round(float(after_ndvi.where(valid).mean()), 4),
        "vegetation_loss_km2": round(n_loss * km2, 2),
        "vegetation_loss_pct": round(100 * n_loss / n_valid, 2),
        "vegetation_gain_km2": round(n_gain * km2, 2),
        "vegetation_gain_pct": round(100 * n_gain / n_valid, 2),
    }


def ndvi_change(
    before: xr.Dataset, after: xr.Dataset, threshold: float, pixel_area_m2: float
) -> ChangeResult:
    # join="exact" raises if the two grids differ at all, instead of silently
    # comparing misaligned pixels.
    before, after = xr.align(before, after, join="exact")
    b, a = ndvi(before), ndvi(after)
    diff = (a - b).rename("ndvi_diff")
    return ChangeResult(b, a, diff, change_stats(b, a, diff, threshold, pixel_area_m2))
