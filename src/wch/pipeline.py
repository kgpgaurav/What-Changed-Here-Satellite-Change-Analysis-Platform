"""Orchestrate the pipeline. Two commands use this module:

`wch run` (10 m, two periods):
    for each period: reuse cached composite if the fingerprint matches, otherwise
        search -> load dates + SCL -> reflectance -> cloud mask -> median composite
    NDVI change + land-cover classes + change types -> GeoTIFFs + summary.json

`wch timeseries` (coarse, all months of several years; needs `wch run` first):
    search -> clearest dates per month -> cloud-masked NDVI stack (cached)
    -> weighted NDVI per zone per date -> monthly series + anomaly flags -> CSV

This module wires steps together and handles I/O; the computations live in
radiometry.py, composite.py, indices.py, change.py, landcover.py, timeseries.py.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import dask
import numpy as np
import rioxarray
import xarray as xr

from wch import catalog
from wch import timeseries as ts
from wch.change import ndvi_change
from wch.composite import clear_mask, median_composite, reflectance_stack
from wch.config import Config, Period
from wch.display import stretch_rgb
from wch.geo import bbox_center, utm_epsg
from wch.indices import ndbi, normalized_difference
from wch.io import read_bands, write_bands, write_json, write_raster
from wch.landcover import change_type, classify, landcover_stats
from wch.load import load_items
from wch.radiometry import radiometry_for_items
from wch.search import (
    median_day_of_year, pick_matched_dates, prefer_collections, rank_dates,
    search_items, select_composite_dates, select_monthly_dates,
)

log = logging.getLogger(__name__)

# Composites whose seasonal centres differ by more than this compare different
# crop stages; results are still written, but with a warning.
MAX_SEASONAL_GAP_DAYS = 10


# --- scene selection (metadata only) ----------------------------------------------

def _search(cfg: Config, datetime_range: str, max_cloud: float, label: str) -> list:
    """Search all collections; keep one collection per date."""
    items = search_items(cfg.stac_url, cfg.collections, cfg.aoi.bbox, datetime_range, max_cloud)
    if not items:
        raise RuntimeError(
            f"No scenes for '{label}' ({datetime_range}) under {max_cloud}% cloud. "
            "Widen the dates or raise max_cloud_cover."
        )
    kept_ids = {d["id"] for d in prefer_collections([i.to_dict() for i in items], cfg.collections)}
    return [i for i in items if i.id in kept_ids]


def select_scenes(cfg: Config) -> dict[str, tuple[list, list[dict]]]:
    """For each period: (all STAC items, chosen date rows). Depends on the method."""
    found = {p.label: _search(cfg, p.stac_datetime, cfg.max_cloud_cover, p.label) for p in cfg.periods}
    dicts = {label: [i.to_dict() for i in items] for label, items in found.items()}
    if cfg.compositing.method == "single":
        b_row, a_row, _ = pick_matched_dates(dicts["before"], dicts["after"], cfg.pair_cloud_limit)
        rows = {"before": [b_row], "after": [a_row]}
    else:
        periods = {p.label: p for p in cfg.periods}
        rows = {
            label: select_composite_dates(d, cfg.compositing.max_scenes,
                                          periods[label].start, periods[label].end)
            for label, d in dicts.items()
        }
    return {label: (found[label], rows[label]) for label in found}


def search_report(cfg: Config) -> dict:
    """What a run would use, without downloading pixels."""
    report = {}
    for label, (items, rows) in select_scenes(cfg).items():
        report[label] = {
            "dates_available": len(rank_dates([i.to_dict() for i in items])),
            "dates_selected": [
                {"date": r["date"], "cloud": r["mean_cloud_cover"], "collection": r["collection"]}
                for r in rows
            ],
            "median_day_of_year": median_day_of_year([r["date"] for r in rows]),
        }
    report["seasonal_gap_days"] = abs(
        report["after"]["median_day_of_year"] - report["before"]["median_day_of_year"]
    )
    report["method"] = cfg.compositing.method
    return report


# --- shared loading -------------------------------------------------------------------

def _load_reflectance(cfg: Config, items: list, rows: list[dict], bands: tuple[str, ...],
                      epsg: int, resolution: int, resampling: dict | None = None):
    """Load the chosen dates and convert to reflectance, deciding the offset per date.

    Returns (reflectance Dataset (time, y, x), SCL DataArray, dates, radiometry).
    """
    chosen_ids = {i for r in rows for i in r["item_ids"]}
    chosen = [i for i in items if i.id in chosen_ids]
    by_id = {i.id: i.to_dict() for i in chosen}
    row_by_date = {r["date"]: r for r in rows}

    raw = load_items(chosen, bands + ("scl",), cfg.aoi.bbox, epsg, resolution, resampling)
    dates = [str(t)[:10] for t in raw.time.values]
    unknown = set(dates) - set(row_by_date)
    if unknown:
        raise RuntimeError(f"loaded dates {sorted(unknown)} don't match the selected dates")

    radiometry = {
        band: [radiometry_for_items([by_id[i] for i in row_by_date[d]["item_ids"]], band) for d in dates]
        for band in bands
    }
    return reflectance_stack(raw, bands, radiometry), raw["scl"], dates, radiometry


# --- compositing -----------------------------------------------------------------------

def _composite_item_id(cfg: Config, period: Period) -> str:
    return f"{cfg.aoi.name}-{period.label}-composite"


def _cached_item(cfg: Config, item_id: str, fingerprint: str):
    """Return the catalog item if it exists, matches the fingerprint and its files exist."""
    item = catalog.find_item(cfg.catalog_dir, item_id)
    if item is None or item.properties.get("wch:fingerprint") != fingerprint:
        return None
    if not all(catalog.asset_path(item, k).exists() for k in item.assets):
        return None
    return item


def _cached(cfg: Config, period: Period):
    return _cached_item(cfg, _composite_item_id(cfg, period), cfg.input_fingerprint())


def build_composite(cfg: Config, period: Period, items: list, rows: list[dict], epsg: int) -> None:
    """Download, mask and composite one period, then write files + catalog item."""
    refl, scl, dates, radiometry = _load_reflectance(cfg, items, rows, cfg.bands, epsg, cfg.resolution)
    row_by_date = {r["date"]: r for r in rows}
    composite, clear_obs = median_composite(refl, clear_mask(scl), cfg.compositing.effective_min_clear_obs)
    log.info("%s: compositing %d dates (downloading now)...", period.label, len(dates))
    # Compute both together so the source data is downloaded once, not twice.
    composite, clear_obs = dask.compute(composite, clear_obs)

    valid_fraction = float(composite["red"].notnull().mean())
    if valid_fraction < 0.9:
        log.warning("%s: only %.0f%% of the area has a trusted value", period.label, 100 * valid_fraction)

    out = cfg.run_dir / "composites"
    refl_path, count_path = out / f"{period.label}_reflectance.tif", out / f"{period.label}_clear_obs.tif"
    write_bands(composite, cfg.bands, refl_path, epsg)
    write_raster(clear_obs, count_path, epsg)

    meta = {
        "window": period.stac_datetime,
        "method": cfg.compositing.method,
        "dates": dates,
        "collections": sorted({row_by_date[d]["collection"] for d in dates}),
        "tile_cloud_cover": [row_by_date[d]["mean_cloud_cover"] for d in dates],
        "median_day_of_year": median_day_of_year(dates),
        "offset_rules": {d: radiometry["red"][i].source for i, d in enumerate(dates)},
        "scene_ids": sorted({i for r in rows for i in r["item_ids"]}),
        "valid_fraction": round(valid_fraction, 4),
        "mean_clear_obs": round(float(clear_obs.mean()), 2),
    }
    item = catalog.make_item(
        _composite_item_id(cfg, period), cfg.aoi, period,
        assets={"reflectance": refl_path, "clear_obs": count_path},
        properties={"wch:fingerprint": cfg.input_fingerprint(), "wch:bands": list(cfg.bands),
                    "wch:epsg": epsg, "wch:summary": meta},
    )
    catalog.upsert_item(cfg.catalog_dir, item)


def get_composites(cfg: Config, epsg: int, force: bool = False) -> dict[str, tuple[xr.Dataset, dict]]:
    """Ensure both composites exist (building only what's missing/stale), then read them.

    Both are always read back from disk, so the two grids come from the same
    source (identical coordinates) whether freshly built or cached.
    """
    stale = [p for p in cfg.periods if force or _cached(cfg, p) is None]
    if stale:
        selection = select_scenes(cfg)
        for period in stale:
            build_composite(cfg, period, *selection[period.label], epsg)
    for period in cfg.periods:
        if period not in stale:
            log.info("%s: reusing cached composite (settings unchanged)", period.label)

    result = {}
    for period in cfg.periods:
        item = _cached(cfg, period)
        result[period.label] = (read_bands(catalog.asset_path(item, "reflectance")),
                                item.properties["wch:summary"])
    return result


# --- full run (10 m change analysis) ----------------------------------------------------

def run(cfg: Config, force: bool = False) -> dict:
    epsg = utm_epsg(*bbox_center(cfg.aoi.bbox))
    out = cfg.run_dir
    out.mkdir(parents=True, exist_ok=True)
    pixel_area = cfg.resolution**2

    comps = get_composites(cfg, epsg, force)
    (before, before_meta), (after, after_meta) = comps["before"], comps["after"]

    gap = abs(after_meta["median_day_of_year"] - before_meta["median_day_of_year"])
    if gap > MAX_SEASONAL_GAP_DAYS:
        log.warning("seasonal centres are %d days apart: crop-stage differences may appear as change", gap)

    # Vegetation change (Stage 1-2)
    result = ndvi_change(before, after, cfg.change_threshold, pixel_area_m2=pixel_area)

    # Land cover and what changed into what (Stage 3)
    lc_before, lc_after = classify(before, cfg.landcover), classify(after, cfg.landcover)
    changes = change_type(lc_before, lc_after)
    ndbi_diff = (ndbi(after) - ndbi(before)).rename("ndbi_diff")
    lc_stats = landcover_stats(lc_before, lc_after, changes, result.ndvi_diff, pixel_area)

    for name, ds in (("before", before), ("after", after)):
        rgb = stretch_rgb(ds["red"], ds["green"], ds["blue"])
        rgb_da = xr.DataArray(
            rgb, dims=("band", "y", "x"),
            coords={"band": [1, 2, 3], "y": ds["y"], "x": ds["x"]},
        )
        write_raster(rgb_da, out / f"{name}_rgb.tif", epsg)
    write_raster(result.before_ndvi.astype(np.float32), out / "before_ndvi.tif", epsg)
    write_raster(result.after_ndvi.astype(np.float32), out / "after_ndvi.tif", epsg)
    write_raster(result.ndvi_diff.astype(np.float32), out / "ndvi_diff.tif", epsg)
    write_raster(ndbi_diff.astype(np.float32), out / "ndbi_diff.tif", epsg)
    write_raster(lc_before, out / "before_landcover.tif", epsg)
    write_raster(lc_after, out / "after_landcover.tif", epsg)
    write_raster(changes, out / "change_type.tif", epsg)

    summary = {
        "aoi": {"name": cfg.aoi.name, "bbox": cfg.aoi.bbox},
        "epsg": epsg,
        "resolution_m": cfg.resolution,
        "compositing": {"method": cfg.compositing.method, "max_scenes": cfg.compositing.max_scenes,
                        "min_clear_obs": cfg.compositing.effective_min_clear_obs},
        "landcover_rules": {"water_mndwi": cfg.landcover.water_mndwi,
                            "vegetation_ndvi": cfg.landcover.vegetation_ndvi,
                            "bare_ndvi": cfg.landcover.bare_ndvi},
        "fingerprint": cfg.input_fingerprint(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasonal_gap_days": gap,
        "periods": {"before": before_meta, "after": after_meta},
        "stats": result.stats,
        "landcover": lc_stats,
    }
    write_json(summary, out / "summary.json")

    change_item = catalog.make_item(
        f"{cfg.aoi.name}-change", cfg.aoi, None,
        assets={name: out / f"{name}.tif" for name in (
            "ndvi_diff", "ndbi_diff", "before_ndvi", "after_ndvi",
            "before_landcover", "after_landcover", "change_type")},
        properties={"wch:fingerprint": cfg.input_fingerprint(), "wch:stats": result.stats,
                    "wch:derived_from": [_composite_item_id(cfg, p) for p in cfg.periods]},
    )
    catalog.upsert_item(cfg.catalog_dir, change_item)
    log.info("wrote results to %s", out)
    return summary


# --- time series (coarse, many dates) ----------------------------------------------------

TS_BANDS = ("red", "nir")


def _timeseries_item_id(cfg: Config) -> str:
    return f"{cfg.aoi.name}-ndvi-timeseries"


def select_timeseries_scenes(cfg: Config) -> tuple[list, list[dict]]:
    t = cfg.timeseries
    items = _search(cfg, f"{t.start_year}-01-01/{t.end_year}-12-31", t.max_cloud_cover, "timeseries")
    rows = select_monthly_dates([i.to_dict() for i in items], t.scenes_per_month)
    return items, rows


def build_ndvi_stack(cfg: Config, epsg: int) -> None:
    """Download red/nir/SCL for the chosen dates at coarse resolution; save NDVI per date."""
    items, rows = select_timeseries_scenes(cfg)
    log.info("timeseries: %d dates between %d and %d at %d m",
             len(rows), cfg.timeseries.start_year, cfg.timeseries.end_year, cfg.timeseries.resolution)
    refl, scl, dates, radiometry = _load_reflectance(
        cfg, items, rows, TS_BANDS, epsg, cfg.timeseries.resolution,
        resampling={"*": "average", "scl": "mode"},
    )
    stack = normalized_difference(refl["nir"], refl["red"]).where(clear_mask(scl))
    log.info("timeseries: downloading (this is the slow part)...")
    stack = stack.compute()

    out = cfg.run_dir / "timeseries"
    stack_path = out / "ndvi_stack.tif"
    write_bands(xr.Dataset({d: stack.isel(time=i, drop=True) for i, d in enumerate(dates)}),
                tuple(dates), stack_path, epsg)
    row_by_date = {r["date"]: r for r in rows}
    item = catalog.make_item(
        _timeseries_item_id(cfg), cfg.aoi, None, assets={"ndvi_stack": stack_path},
        properties={"wch:fingerprint": cfg.timeseries_fingerprint(), "wch:epsg": epsg,
                    "wch:date_collections": {d: row_by_date[d]["collection"] for d in dates},
                    "wch:offset_rules": {d: radiometry["red"][i].source for i, d in enumerate(dates)}},
    )
    catalog.upsert_item(cfg.catalog_dir, item)


def run_timeseries(cfg: Config, force: bool = False) -> dict:
    epsg = utm_epsg(*bbox_center(cfg.aoi.bbox))
    out = cfg.run_dir
    lc_paths = [out / "before_landcover.tif", out / "after_landcover.tif"]
    if not all(p.exists() for p in lc_paths):
        raise RuntimeError("Zones come from the change map: run `wch run --config ...` first.")

    fingerprint = cfg.timeseries_fingerprint()
    item = None if force else _cached_item(cfg, _timeseries_item_id(cfg), fingerprint)
    if item is None:
        build_ndvi_stack(cfg, epsg)
        item = _cached_item(cfg, _timeseries_item_id(cfg), fingerprint)
    else:
        log.info("timeseries: reusing cached NDVI stack (settings unchanged)")

    stack_ds = read_bands(catalog.asset_path(item, "ndvi_stack"))
    dates = list(stack_ds.data_vars)
    stack = np.stack([stack_ds[d].values for d in dates])
    template = stack_ds[dates[0]]

    lc_before, lc_after = (rioxarray.open_rasterio(p).isel(band=0, drop=True) for p in lc_paths)
    weights = ts.zone_weights(ts.zone_masks(lc_before, lc_after), template, epsg)
    per_date = ts.zone_series(stack, dates, weights, cfg.timeseries.min_zone_clear_fraction)
    collections = item.properties.get("wch:date_collections", {})
    per_date["collection"] = per_date["date"].map(collections)

    monthly = ts.flag_anomalies(ts.monthly(per_date), cfg.timeseries.anomaly_z)
    months = sorted({m for p in cfg.periods for m in range(p.start.month, p.end.month + 1)})
    season = ts.season_summary(monthly, months)

    ts_dir = out / "timeseries"
    per_date.to_csv(ts_dir / "per_date.csv", index=False)
    monthly.to_csv(ts_dir / "monthly.csv", index=False)
    season.to_csv(ts_dir / "season_by_year.csv")
    anomalies = monthly[monthly["is_anomaly"]]
    result = {
        "n_dates": len(dates),
        "first_date": dates[0], "last_date": dates[-1],
        "season_months": months,
        "n_anomalies": int(len(anomalies)),
        "anomalies": anomalies[["zone", "year", "month", "ndvi", "typical", "robust_z"]].to_dict(orient="records"),
    }
    write_json(result, ts_dir / "timeseries_summary.json")
    log.info("timeseries: %d dates, %d anomalous months; wrote %s", len(dates), len(anomalies), ts_dir)
    return result