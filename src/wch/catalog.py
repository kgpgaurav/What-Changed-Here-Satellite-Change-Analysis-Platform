"""A small local STAC catalog describing what the pipeline produced.

Why: every output file gets a standard, machine-readable record of *what* it is
(area, time window, bands), *where it came from* (source scene IDs and dates)
and *which settings made it* (the config fingerprint). That gives provenance,
and lets the pipeline skip rebuilding a composite whose inputs haven't changed.

Layout on disk (plain JSON files, readable in any text editor or QGIS STAC tools):
    data/catalog/catalog.json
    data/catalog/<item-id>/<item-id>.json
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pystac

from wch.config import AOI, Period


def _polygon(bbox) -> dict:
    min_lon, min_lat, max_lon, max_lat = bbox
    ring = [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat],
            [min_lon, max_lat], [min_lon, min_lat]]
    return {"type": "Polygon", "coordinates": [ring]}


def _as_datetime(d) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def open_catalog(root: Path) -> pystac.Catalog:
    path = Path(root) / "catalog.json"
    if path.exists():
        return pystac.Catalog.from_file(str(path))
    return pystac.Catalog(id="what-changed-here", description="Composites and change products")


def find_item(root: Path, item_id: str) -> pystac.Item | None:
    if not (Path(root) / "catalog.json").exists():
        return None
    return next(open_catalog(root).get_items(item_id), None)


def upsert_item(root: Path, item: pystac.Item) -> None:
    """Add the item, replacing any previous version with the same id, and save."""
    root = Path(root)
    catalog = open_catalog(root)
    if next(catalog.get_items(item.id), None) is not None:
        catalog.remove_item(item.id)
    catalog.add_item(item)
    catalog.normalize_hrefs(str(root))
    catalog.make_all_asset_hrefs_relative()  # the whole data/ folder can be moved or zipped
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)


def asset_path(item: pystac.Item, key: str) -> Path:
    return Path(item.assets[key].get_absolute_href())


def make_item(
    item_id: str,
    aoi: AOI,
    period: Period | None,
    assets: dict[str, Path],
    properties: dict,
) -> pystac.Item:
    """A STAC item for one output. Composites cover a time *range*, so we use
    start/end datetimes instead of a single datetime."""
    start = _as_datetime(period.start) if period else None
    end = _as_datetime(period.end) if period else None
    item = pystac.Item(
        id=item_id,
        geometry=_polygon(aoi.bbox),
        bbox=list(aoi.bbox),
        datetime=None if period else datetime.now(timezone.utc),
        start_datetime=start,
        end_datetime=end,
        properties={"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **properties},
    )
    for key, path in assets.items():
        item.add_asset(key, pystac.Asset(
            href=str(Path(path).resolve()), media_type=pystac.MediaType.GEOTIFF, roles=["data"],
        ))
    return item