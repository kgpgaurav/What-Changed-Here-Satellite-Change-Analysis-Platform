"""Temporal analysis: monthly NDVI per zone, 2019-2026, with anomaly flags.

Why: comparing two composites cannot tell you whether a difference is a lasting
change or just an unusual season. A monthly series shows the normal crop cycle,
whether the "before" or "after" year was unusual, and *when* land in the
"vegetation -> built-up/bare" zone stopped behaving like farmland.

Zones come from the 10 m change map (see landcover.py):
    whole area | stable vegetation | stable built-up/bare | vegetation -> built-up/bare
The series is loaded at coarse resolution (default 100 m), so each coarse pixel
gets a *weight* = the fraction of its area that belongs to the zone.

Anomalies: for each zone and calendar month (e.g. every February), compare each
year with the median of all years. The spread is measured with the MAD (median
absolute deviation), which, unlike a standard deviation, is not inflated by the
very outliers we are trying to find. robust z = (x - median) / (1.4826 * MAD).
This is a statistical rule, not ML; the ML component is Stage 4.

Anomalies are only computed for the two *stable* zones. The method assumes a
zone's normal behaviour doesn't change over the years; where land was built on
partway through, every pre-construction year would look "unusual" against the
multi-year median. In changed zones the series itself shows the change instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr
from rasterio.enums import Resampling

from wch.landcover import BUILT, VEG

ZONES = ("whole area", "stable vegetation", "stable built-up/bare", "vegetation -> built-up/bare")
ANOMALY_ZONES = ("stable vegetation", "stable built-up/bare")
MAD_TO_SIGMA = 1.4826
MIN_SPREAD = 0.02   # NDVI; stops tiny MADs (very regular months) from flagging noise
MIN_YEARS = 4       # need this many years of a calendar month to judge "unusual"


def zone_masks(before_lc: xr.DataArray, after_lc: xr.DataArray) -> dict[str, xr.DataArray]:
    """10 m boolean masks for each zone."""
    valid = (before_lc > 0) & (after_lc > 0)
    return {
        "whole area": valid,
        "stable vegetation": (before_lc == VEG) & (after_lc == VEG),
        "stable built-up/bare": (before_lc == BUILT) & (after_lc == BUILT),
        "vegetation -> built-up/bare": (before_lc == VEG) & (after_lc == BUILT),
    }


def zone_weights(masks: dict[str, xr.DataArray], template: xr.DataArray, epsg: int) -> dict[str, np.ndarray]:
    """Resample each 10 m mask onto the coarse grid as an area fraction (0..1)."""
    template = template.rio.write_crs(f"EPSG:{epsg}")
    out = {}
    for name, mask in masks.items():
        frac = mask.astype("float32").rio.write_crs(f"EPSG:{epsg}").rio.write_nodata(np.nan)
        frac = frac.rio.reproject_match(template, resampling=Resampling.average)
        out[name] = np.nan_to_num(frac.values)
    return out


def zone_series(
    ndvi_stack: np.ndarray, dates: list[str], weights: dict[str, np.ndarray], min_clear: float
) -> pd.DataFrame:
    """Weighted mean NDVI per zone per date.

    ndvi_stack: (time, y, x), NaN where cloudy/no data. A zone's value on a date is
    dropped if less than `min_clear` of its (weighted) area was cloud-free.
    """
    valid = ~np.isnan(ndvi_stack)
    filled = np.nan_to_num(ndvi_stack)
    rows = []
    for zone, w in weights.items():
        total = w.sum()
        if total <= 0:
            continue
        wv = valid * w
        clear_w = wv.sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (filled * wv).sum(axis=(1, 2)) / clear_w
        clear_frac = clear_w / total
        mean = np.where(clear_frac >= min_clear, mean, np.nan)
        rows += [{"date": d, "zone": zone, "ndvi": float(m), "clear_fraction": round(float(c), 3)}
                 for d, m, c in zip(dates, mean, clear_frac)]
    return pd.DataFrame(rows)


def monthly(per_date: pd.DataFrame) -> pd.DataFrame:
    """Median of the dates in each month, per zone."""
    df = per_date.dropna(subset=["ndvi"]).copy()
    dt = pd.to_datetime(df["date"])
    df["year"], df["month"] = dt.dt.year, dt.dt.month
    out = (df.groupby(["zone", "year", "month"])["ndvi"]
             .agg(ndvi="median", n_dates="count").reset_index())
    out["ndvi"] = out["ndvi"].round(4)
    return out


def flag_anomalies(monthly_df: pd.DataFrame, z_threshold: float,
                   zones: tuple[str, ...] = ANOMALY_ZONES) -> pd.DataFrame:
    """Add typical (median across years), robust z and is_anomaly per zone+calendar month.
    robust_z is left empty for zones not in `zones` (see module docstring)."""
    df = monthly_df.copy()
    grp = df.groupby(["zone", "month"])["ndvi"]
    df["typical"] = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    n_years = grp.transform("count")
    spread = np.maximum(MAD_TO_SIGMA * mad, MIN_SPREAD)
    usable = (n_years >= MIN_YEARS) & df["zone"].isin(zones)
    df["robust_z"] = ((df["ndvi"] - df["typical"]) / spread).where(usable).round(2)
    df["is_anomaly"] = df["robust_z"].abs() > z_threshold
    df["typical"] = df["typical"].round(4)
    return df


def season_summary(monthly_df: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Mean NDVI per zone per year over the analysis months (e.g. Feb-Mar):
    shows whether the 'before' or 'after' year was unusual compared with the others."""
    df = monthly_df[monthly_df["month"].isin(months)]
    table = df.groupby(["zone", "year"])["ndvi"].mean().round(3).unstack("year")
    return table.reindex([z for z in ZONES if z in table.index])