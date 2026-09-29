"""Cloud masking and compositing: many noisy images -> one clean image per period.

All functions are pure (arrays in, arrays out) and work lazily with dask, so
the same code runs on a 3x3 test array and on a 2400x2000x8 real stack.
"""
from __future__ import annotations

import xarray as xr

from wch.radiometry import Radiometry, to_reflectance

# Sentinel-2 Scene Classification (SCL) codes produced by ESA's Sen2Cor processor.
SCL_CLASSES = {
    0: "no data", 1: "saturated/defective", 2: "dark area / topographic shadow",
    3: "cloud shadow", 4: "vegetation", 5: "not vegetated", 6: "water",
    7: "unclassified", 8: "cloud (medium probability)", 9: "cloud (high probability)",
    10: "thin cirrus", 11: "snow/ice",
}
# Classes we do not trust as a view of the ground.
SCL_MASKED = (0, 1, 2, 3, 8, 9, 10)


def clear_mask(scl: xr.DataArray) -> xr.DataArray:
    """True where the pixel is a usable, cloud-free observation."""
    return ~scl.isin(SCL_MASKED)


def reflectance_stack(
    raw: xr.Dataset, bands: tuple[str, ...], radiometry: dict[str, list[Radiometry]]
) -> xr.Dataset:
    """Convert each date with its own scale/offset.

    Dates within one period can come from different collections or processing
    baselines, so the offset rule is decided per date (radiometry[band][i] is
    for raw.time[i]).
    """
    out = {}
    for band in bands:
        slices = [
            to_reflectance(raw[band].isel(time=i), rad)
            for i, rad in enumerate(radiometry[band])
        ]
        out[band] = xr.concat(slices, dim="time")
    return xr.Dataset(out)


def median_composite(
    refl: xr.Dataset, clear: xr.DataArray, min_clear_obs: int
) -> tuple[xr.Dataset, xr.DataArray]:
    """Per-pixel median of the clear observations.

    The median is robust: a cloud edge the mask missed is a bright outlier and
    barely moves the median (a mean would be pulled by it).
    Returns (composite with dims (y, x), number of clear observations per pixel).
    """
    masked = refl.where(clear)
    clear_obs = masked["red"].notnull().sum("time").astype("uint8").rename("clear_obs")
    composite = masked.median("time", skipna=True).where(clear_obs >= min_clear_obs)
    return composite, clear_obs