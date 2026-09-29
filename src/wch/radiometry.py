"""Convert raw Sentinel-2 pixel values (digital numbers, DN) to surface reflectance.

Reflectance = fraction of sunlight reflected, roughly 0-1. Files store it as
integers: reflectance = DN * 0.0001 + offset.

The catch: from processing baseline 04.00 (25 Jan 2022) ESA added an offset of
-1000 DN (-0.1 reflectance). Older scenes have no offset. NDVI is a ratio of
differences and sums, so an un-removed additive offset biases it, which would
create fake "change" between a 2019 and a 2026 scene.

Where the offset comes from, in order of trust:
  1. `earthsearch:boa_offset_applied: true` -> provider already removed it.
  2. STAC `raster:bands` (STAC 1.0) or `bands` (STAC 1.1) scale/offset metadata.
  3. Fallback rule: baseline >= 04.00 -> offset -0.1, else 0.
The chosen rule is recorded per period in summary.json so you can audit it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import xarray as xr

DEFAULT_SCALE = 1e-4
POST_2022_OFFSET = -0.1  # -1000 DN * 1e-4


@dataclass(frozen=True)
class Radiometry:
    scale: float
    offset: float
    source: str  # human-readable reason, stored in the run summary

    def as_dict(self) -> dict:
        return asdict(self)


def _band_metadata(item: dict, asset_key: str) -> dict | None:
    asset = item.get("assets", {}).get(asset_key, {})
    bands = asset.get("raster:bands") or asset.get("bands")
    if isinstance(bands, list) and bands and isinstance(bands[0], dict):
        return bands[0]
    return None


def processing_baseline(item: dict) -> float | None:
    value = item.get("properties", {}).get("s2:processing_baseline")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def radiometry_for(item: dict, asset_key: str) -> Radiometry:
    """Decide scale/offset for one band of one STAC item (given as a dict)."""
    if item.get("properties", {}).get("earthsearch:boa_offset_applied") is True:
        return Radiometry(DEFAULT_SCALE, 0.0, "provider already removed offset")

    meta = _band_metadata(item, asset_key)
    if meta and ("scale" in meta or "offset" in meta):
        return Radiometry(
            float(meta.get("scale", DEFAULT_SCALE)),
            float(meta.get("offset", 0.0)),
            "STAC band metadata",
        )

    baseline = processing_baseline(item)
    if baseline is not None and baseline >= 4.0:
        return Radiometry(DEFAULT_SCALE, POST_2022_OFFSET, f"baseline {baseline:.2f} >= 04.00 rule")
    return Radiometry(DEFAULT_SCALE, 0.0, f"baseline {baseline} < 04.00 rule (no offset)")


def radiometry_for_items(items: list[dict], asset_key: str) -> Radiometry:
    """All tiles mosaicked into one image must share the same conversion."""
    found = {radiometry_for(item, asset_key) for item in items}
    values = {(r.scale, r.offset) for r in found}
    if len(values) != 1:
        raise ValueError(f"tiles disagree on radiometry for '{asset_key}': {found}")
    return next(iter(found))


def to_reflectance(dn: xr.DataArray, rad: Radiometry) -> xr.DataArray:
    """DN -> reflectance. DN 0 means 'no data' (outside the tile) and becomes NaN.

    Small negative values can appear after the offset over very dark surfaces
    (e.g. water); we clip them to 0 so ratios like NDVI stay well-behaved.
    """
    refl = dn.astype("float32") * rad.scale + rad.offset
    return refl.where(dn != 0).clip(min=0.0)
