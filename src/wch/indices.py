"""Spectral indices computed from reflectance bands.

A normalized difference (a - b) / (a + b) compares two bands on a -1..1 scale
and cancels out overall brightness (e.g. a slightly hazier day).

NDVI  = (NIR - Red)   / (NIR + Red)    vegetation: plants reflect NIR, absorb red.
        ~0.5-0.9 crops/trees, ~0.1-0.2 soil and buildings, < 0 water.
NDBI  = (SWIR - NIR)  / (SWIR + NIR)   built-up: concrete/roofs/soil reflect more
        short-wave infrared than near-infrared, vegetation the opposite. Bare soil
        also scores high, so NDBI alone cannot separate buildings from dry fields.
MNDWI = (Green - SWIR) / (Green + SWIR) water: water absorbs SWIR strongly, so
        it is clearly positive over rivers and ponds.
"""
from __future__ import annotations

import xarray as xr


def normalized_difference(a: xr.DataArray, b: xr.DataArray, eps: float = 1e-6) -> xr.DataArray:
    denom = a + b
    return ((a - b) / denom).where(abs(denom) > eps)


def ndvi(ds: xr.Dataset) -> xr.DataArray:
    return normalized_difference(ds["nir"], ds["red"]).rename("ndvi")


def ndbi(ds: xr.Dataset) -> xr.DataArray:
    return normalized_difference(ds["swir16"], ds["nir"]).rename("ndbi")


def mndwi(ds: xr.Dataset) -> xr.DataArray:
    return normalized_difference(ds["green"], ds["swir16"]).rename("mndwi")