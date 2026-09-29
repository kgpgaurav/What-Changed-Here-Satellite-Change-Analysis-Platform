"""Find Sentinel-2 scenes via a STAC API and pick which ones to use.

A STAC *item* is a JSON record for one scene (one 110x110 km tile on one date):
its date, cloud cover, footprint and *assets* (links to one file per band).
Searching returns only this metadata; no pixels are downloaded here.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date

from pystac import Item
from pystac_client import Client

log = logging.getLogger(__name__)


def search_items(
    stac_url: str,
    collections: tuple[str, ...],
    bbox: tuple[float, float, float, float],
    datetime_range: str,
    max_cloud_cover: float,
) -> list[Item]:
    """Search every collection and return all items (merged later by `prefer_collections`)."""
    client = Client.open(stac_url)
    items: list[Item] = []
    for collection in collections:
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        found = list(search.items())
        log.info("%s %s: %d items", collection, datetime_range, len(found))
        items.extend(found)
    return items


def item_date(item: dict) -> str:
    """Acquisition date YYYY-MM-DD. Over India the overpass is ~05:30 UTC, so UTC date = local date."""
    return item["properties"]["datetime"][:10]


def prefer_collections(item_dicts: list[dict], collection_order: tuple[str, ...]) -> list[dict]:
    """For each date keep items from the highest-priority collection that has that date.

    Both collections can contain the same acquisition, processed differently
    (e.g. different offset conventions). Taking one collection per date means a
    single image never mixes the two.
    """
    rank = {c: i for i, c in enumerate(collection_order)}
    by_date: dict[str, list[dict]] = defaultdict(list)
    for item in item_dicts:
        by_date[item_date(item)].append(item)
    kept = []
    for items in by_date.values():
        best = min(rank.get(i.get("collection"), len(rank)) for i in items)
        kept.extend(i for i in items if rank.get(i.get("collection"), len(rank)) == best)
    return kept


def group_by_date(item_dicts: list[dict]) -> dict[str, list[dict]]:
    """Group items by date. Our area can straddle two tiles; same-date tiles come
    from the same satellite pass, so they are mosaicked into one image."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in item_dicts:
        groups[item_date(item)].append(item)
    return dict(groups)


def _mean_cloud(items: list[dict]) -> float:
    return sum(i["properties"].get("eo:cloud_cover", 100.0) for i in items) / len(items)


def rank_dates(item_dicts: list[dict]) -> list[dict]:
    """Candidate dates, best first: lowest mean cloud cover, then most tiles."""
    rows = [
        {"date": d, "n_items": len(items), "mean_cloud_cover": round(_mean_cloud(items), 2),
         "collection": items[0].get("collection"), "item_ids": [i["id"] for i in items]}
        for d, items in group_by_date(item_dicts).items()
    ]
    return sorted(rows, key=lambda r: (r["mean_cloud_cover"], -r["n_items"], r["date"]))


def day_of_year_gap(d1: str, d2: str) -> int:
    """Days between two dates ignoring the year (wraps around New Year)."""
    gap = abs(date.fromisoformat(d1).timetuple().tm_yday - date.fromisoformat(d2).timetuple().tm_yday)
    return min(gap, 365 - gap)


def pick_matched_dates(
    before_dicts: list[dict], after_dicts: list[dict], cloud_limit: float
) -> tuple[dict, dict, int]:
    """Stage 1 strategy: one date per period, matched by season.

    Crops look very different in early February vs late March, so comparing
    mismatched dates shows crop growth, not land-use change. Among dates with
    cloud cover <= cloud_limit, pick the before/after pair closest in
    day-of-year (ties: lower total cloud). If a period has no date under the
    limit, all its dates are considered.
    Returns (before_row, after_row, gap_days); rows come from `rank_dates`.
    """
    if not before_dicts or not after_dicts:
        raise ValueError("both periods need at least one item")

    def candidates(dicts):
        rows = rank_dates(dicts)
        clear = [r for r in rows if r["mean_cloud_cover"] <= cloud_limit]
        return clear or rows

    pairs = [
        (day_of_year_gap(b["date"], a["date"]), b["mean_cloud_cover"] + a["mean_cloud_cover"], b, a)
        for b in candidates(before_dicts)
        for a in candidates(after_dicts)
    ]
    gap, _, before, after = min(pairs, key=lambda p: (p[0], p[1], p[2]["date"], p[3]["date"]))
    return before, after, gap


def select_composite_dates(
    item_dicts: list[dict], max_scenes: int, start: date, end: date
) -> list[dict]:
    """Stage 2 strategy: dates spread evenly across the window, clearest first.

    Picking simply "the N clearest dates" can bunch them into one part of the
    season (e.g. all of February in one year, mostly March in the other), and
    then the two composites show different crop stages. Instead we split the
    window into `max_scenes` equal time slices and take the clearest date in each.
    Slices without any date are filled with the clearest remaining dates.
    Both periods use the same months, so their composites share a seasonal centre.
    Returns rows from `rank_dates`, in date order.
    """
    if not item_dicts:
        raise ValueError("no items to choose from")
    rows = rank_dates(item_dicts)  # clearest first
    span = (end - start).days + 1

    def offset(row) -> int:
        return (date.fromisoformat(row["date"]) - start).days

    chosen: list[dict] = []
    for k in range(max_scenes):
        lo, hi = span * k // max_scenes, span * (k + 1) // max_scenes
        in_slice = [r for r in rows if lo <= offset(r) < hi and r not in chosen]
        if in_slice:
            chosen.append(in_slice[0])
    for r in rows:  # fill empty slices
        if len(chosen) >= max_scenes:
            break
        if r not in chosen:
            chosen.append(r)
    return sorted(chosen, key=lambda r: r["date"])


def median_day_of_year(dates: list[str]) -> int:
    """Seasonal 'centre' of a set of dates. Comparing two composites is fair only
    if their centres are close (same crop stage)."""
    days = sorted(date.fromisoformat(d).timetuple().tm_yday for d in dates)
    return days[len(days) // 2]


def select_monthly_dates(item_dicts: list[dict], per_month: int) -> list[dict]:
    """Time-series strategy: the `per_month` clearest dates in every calendar month.

    A fixed number per month keeps the series evenly sampled (cloudy monsoon
    months simply contribute fewer dates). Returns rows from `rank_dates`, in date order.
    """
    by_month: dict[str, list[dict]] = defaultdict(list)
    for row in rank_dates(item_dicts):          # clearest first
        by_month[row["date"][:7]].append(row)   # "YYYY-MM"
    chosen = [r for rows in by_month.values() for r in rows[:per_month]]
    return sorted(chosen, key=lambda r: r["date"])