"""Load pixels for STAC items into an xarray Dataset on a fixed grid.

odc-stac reads the Cloud-Optimized GeoTIFFs over HTTP, fetching only the parts
that overlap our bbox, and places them onto one output grid we define
(CRS + resolution + bbox). Loading both periods with identical grid parameters
is what guarantees that pixel [i, j] is the same 10 m square on the ground in
2019 and 2026.

Resampling: the 10 m bands already sit on a 10 m grid. The SCL cloud layer is
20 m, so each SCL pixel is copied into four 10 m pixels. It must use "nearest":
averaging class codes (e.g. 4 = vegetation, 8 = cloud -> 6 = "water") is meaningless.

The result is *lazy* (backed by dask): nothing is downloaded until .compute().
"""
from __future__ import annotations

import logging

import odc.stac
import xarray as xr
from pystac import Item

log = logging.getLogger(__name__)

_configured = False


def _configure_once() -> None:
    """Sensible GDAL defaults for reading public cloud-hosted files (no credentials)."""
    global _configured
    if not _configured and hasattr(odc.stac, "configure_rio"):
        odc.stac.configure_rio(cloud_defaults=True)
        _configured = True


def load_items(
    items: list[Item],
    bands: tuple[str, ...],
    bbox: tuple[float, float, float, float],
    epsg: int,
    resolution: int,
    resampling: dict[str, str] | None = None,
) -> xr.Dataset:
    """Load items as a lazy Dataset of raw values with dims (time, y, x).

    Tiles from the same date are mosaicked into one time step. The default
    resampling suits the 10 m grid; the coarse time series passes its own
    ("average" reflectance, "mode" = most common class for SCL).
    """
    _configure_once()
    ds = odc.stac.load(
        items,
        bands=list(bands),
        bbox=list(bbox),
        crs=f"EPSG:{epsg}",
        resolution=resolution,
        groupby="solar_day",                              # one time step per satellite pass
        chunks={"x": 1024, "y": 1024},                    # process in blocks to limit memory
        resampling=resampling or {"*": "bilinear", "scl": "nearest"},  # never average categories
    )
    log.info(
        "grid: %d x %d pixels at %d m (EPSG:%d), %d dates",
        ds.sizes["x"], ds.sizes["y"], resolution, epsg, ds.sizes["time"],
    )
    # A per-pixel median needs all dates of that pixel in the same block.
    return ds.chunk({"time": -1})