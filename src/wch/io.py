"""Write and read results on disk.

GeoTIFF = an image plus georeferencing (CRS + pixel-to-map transform), so any
GIS tool (QGIS, rasterio, our app) knows exactly where each pixel sits.
rioxarray's `.rio` accessor handles that bookkeeping for xarray objects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr


def write_raster(da: xr.DataArray, path: Path, epsg: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    da = da.rio.write_crs(f"EPSG:{epsg}")
    nodata = np.nan if np.issubdtype(da.dtype, np.floating) else 0
    da = da.rio.write_nodata(nodata)
    da.rio.to_raster(path, compress="deflate", tiled=True)


def write_bands(ds: xr.Dataset, bands: tuple[str, ...], path: Path, epsg: int) -> None:
    """Several float bands -> one multi-band GeoTIFF; band names stored as descriptions."""
    stack = xr.concat([ds[b].astype("float32") for b in bands], dim="band")
    stack = stack.assign_coords(band=list(range(1, len(bands) + 1)))
    stack.attrs["long_name"] = tuple(bands)
    write_raster(stack, path, epsg)


def read_bands(path: Path) -> xr.Dataset:
    """Inverse of write_bands: returns a Dataset with one variable per band, NaN = no data."""
    da = rioxarray.open_rasterio(path, masked=True)
    names = da.attrs["long_name"]
    names = [names] if isinstance(names, str) else list(names)
    out = {}
    for i, name in enumerate(names):
        band = da.isel(band=i, drop=True)
        # Drop file-level attributes (like the band-name list) so they don't leak
        # into arrays derived from this band and confuse later writes.
        band.attrs = {}
        out[name] = band
    return xr.Dataset(out)


def write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))