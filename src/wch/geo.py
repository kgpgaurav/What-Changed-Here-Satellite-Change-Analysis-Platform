"""Small geographic helpers.

Why UTM? Latitude/longitude degrees are not equal-length on the ground, so a
"pixel" in lon/lat is not a fixed area. In a UTM projection pixels are true
metre squares, so area = pixel_count * resolution**2. Each UTM zone is 6 degrees
of longitude wide; Kanpur (80.3 E) is in zone 44 North -> EPSG:32644.
"""
from __future__ import annotations

import math


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 / UTM zone containing (lon, lat)."""
    zone = int((lon + 180) // 6) + 1
    zone = min(zone, 60)  # lon == 180 edge case
    return (32600 if lat >= 0 else 32700) + zone


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (min_lon + max_lon) / 2, (min_lat + max_lat) / 2


def bbox_size_km(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Approximate (width, height) of a lon/lat box in km. Good enough for sanity checks."""
    min_lon, min_lat, max_lon, max_lat = bbox
    km_per_deg_lat = 110.95
    mid_lat = math.radians((min_lat + max_lat) / 2)
    width = (max_lon - min_lon) * 111.32 * math.cos(mid_lat)
    height = (max_lat - min_lat) * km_per_deg_lat
    return width, height
