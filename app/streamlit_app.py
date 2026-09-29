"""Streamlit viewer: shows the files written by `wch run`, `wch timeseries` and
`wch train`, and has a Label tab for creating the classifier's training labels.

Run from the project root:  streamlit run app/streamlit_app.py

The app never computes anything heavy itself; it only reads pipeline outputs.
That keeps the analysis reproducible from the command line.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import streamlit as st
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from streamlit_folium import st_folium

from wch.display import colorize, colorize_classes
from wch.modeling import append_label, labels_path, points_path, read_labels, undo_last_label

DATA_DIR = Path(os.environ.get("WCH_DATA_DIR", "data/processed"))
LABELS_DIR = Path(os.environ.get("WCH_LABELS_DIR", "labels"))   # must match `labels.dir` in the config
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
CHIP_PX, CHIP_ZOOM = 48, 6        # label chips: 48 pixels = 480 m, shown 6x enlarged
MAX_DISPLAY_PX = 1600  # downsample big rasters so the browser stays responsive

LANDCOVER_PALETTE = {1: "#2c7fb8", 2: "#d95f0e", 3: "#31a354", 4: "#c2e699"}
LANDCOVER_LEGEND = {"water": "#2c7fb8", "built-up/bare": "#d95f0e",
                    "sparse vegetation": "#c2e699", "vegetation": "#31a354"}
CHANGE_PALETTE = {1: "#c7e9c0", 2: "#d9d9d9", 3: "#c6dbef", 4: "#e31a1c", 5: "#1b7837",
                  6: "#2171b5", 7: "#fff7bc"}
CHANGE_LEGEND = {
    "vegetation → built-up/bare": "#e31a1c", "built-up/bare → vegetation": "#1b7837",
    "water change": "#2171b5", "stable vegetation": "#c7e9c0",
    "stable built-up/bare": "#d9d9d9", "stable water": "#c6dbef",
    "sparse vegetation (either year)": "#fff7bc",
}


def layer_specs(max_scenes: int) -> dict:
    """name -> (file relative to the run folder, kind, value range or palette)."""
    return {
        "Before: true colour": ("before_rgb.tif", "rgb", None),
        "After: true colour": ("after_rgb.tif", "rgb", None),
        "Change type": ("change_type.tif", "classes", CHANGE_PALETTE),
        "Before: land cover": ("before_landcover.tif", "classes", LANDCOVER_PALETTE),
        "After: land cover": ("after_landcover.tif", "classes", LANDCOVER_PALETTE),
        "NDVI change": ("ndvi_diff.tif", "RdBu", (-0.5, 0.5)),
        "Built-up index (NDBI) change": ("ndbi_diff.tif", "PuOr_r", (-0.3, 0.3)),
        "Before: NDVI": ("before_ndvi.tif", "RdYlGn", (-0.1, 0.8)),
        "After: NDVI": ("after_ndvi.tif", "RdYlGn", (-0.1, 0.8)),
        "Before: clear observations": ("composites/before_clear_obs.tif", "viridis", (0, max_scenes)),
        "After: clear observations": ("composites/after_clear_obs.tif", "viridis", (0, max_scenes)),
        "Model: predicted construction": ("model/construction.tif", "classes", {1: "#ff00ff"}),
        "Model: construction probability": ("model/construction_probability.tif", "magma", (0.0, 1.0)),
    }


@st.cache_data(show_spinner=False)
def load_overlay(path: str, kind: str, param, mtime: float):
    """Read a GeoTIFF, reproject to lon/lat for the web map, return (RGBA array, bounds).

    `mtime` is only part of the cache key, so re-running the pipeline refreshes the image.
    Class maps are reprojected with "nearest" so codes are never blended.
    """
    da = rioxarray.open_rasterio(path, masked=kind != "classes")
    nearest = kind in ("classes", "rgb")
    da = da.rio.reproject("EPSG:4326", resampling=Resampling.nearest if nearest else Resampling.bilinear)
    step = max(1, math.ceil(max(da.sizes["x"], da.sizes["y"]) / MAX_DISPLAY_PX))
    da = da.isel(x=slice(None, None, step), y=slice(None, None, step))
    arr = da.values
    if kind == "rgb":
        rgb = np.nan_to_num(arr).astype("uint8").transpose(1, 2, 0)
        alpha = np.where(np.isnan(arr).any(axis=0) | (rgb.sum(axis=2) == 0), 0, 255)
        rgba = np.dstack([rgb, alpha.astype("uint8")])
    elif kind == "classes":
        rgba = colorize_classes(arr[0], dict(param))
    else:
        rgba = colorize(arr[0], kind, *param)
    minx, miny, maxx, maxy = da.rio.bounds()
    return rgba, [[miny, minx], [maxy, maxx]]


def build_map(run_dir: Path, layers: dict, selected: list[str], opacity: float, bbox) -> folium.Map:
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    folium.TileLayer(
        ESRI_URL, attr="Esri World Imagery", name="Current high-res imagery",
    ).add_to(m)
    for name in selected:
        file, kind, param = layers[name]
        path = run_dir / file
        if not path.exists():
            continue
        hashable = tuple(sorted(param.items())) if isinstance(param, dict) else param
        rgba, bounds = load_overlay(str(path), kind, hashable, path.stat().st_mtime)
        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=bounds, opacity=opacity, name=name, mercator_project=True,
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def legend(entries: dict[str, str]) -> None:
    html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="width:12px;height:12px;background:{c};display:inline-block;border:1px solid #999"></span>'
        f'{name}</div>' for name, c in entries.items())
    st.sidebar.markdown(html, unsafe_allow_html=True)


def scenes_table(summary: dict) -> pd.DataFrame:
    rows = []
    for label, p in summary["periods"].items():
        dates = p.get("dates", [])
        rows.append({
            "period": label,
            "dates used": len(dates),
            "first / last date": f"{dates[0]} / {dates[-1]}" if dates else "",
            "seasonal centre (day of year)": p.get("median_day_of_year"),
            "collections": ", ".join(p.get("collections", [])),
            "offset rules": "; ".join(sorted(set(p.get("offset_rules", {}).values()))),
            "avg clear obs per pixel": p.get("mean_clear_obs"),
            "area with trusted data %": round(100 * p.get("valid_fraction", 0), 1),
        })
    return pd.DataFrame(rows)


def tab_map(run_dir: Path, summary: dict) -> None:
    comp = summary.get("compositing", {"method": "single", "max_scenes": 1})
    layers = layer_specs(comp["max_scenes"])
    selected = st.sidebar.multiselect(
        "Map layers", list(layers), default=["Before: true colour", "After: true colour", "Change type"]
    )
    opacity = st.sidebar.slider("Layer opacity", 0.2, 1.0, 0.85, 0.05)
    st.sidebar.caption("Change type colours")
    legend(CHANGE_LEGEND)
    st.sidebar.caption("Land cover colours")
    legend(LANDCOVER_LEGEND)
    st.sidebar.caption("NDVI change: red = less green, blue = greener. "
                       "Clear observations: yellow = many cloud-free dates, purple = few. "
                       "Model layers (after `wch train`): magenta = predicted new construction; "
                       "probability from dark (0) to bright (1).")
    m = build_map(run_dir, layers, selected, opacity, summary["aoi"]["bbox"])
    st_folium(m, height=620, width=None, returned_objects=[])


def tab_changes(summary: dict) -> None:
    lc = summary.get("landcover")
    if not lc:
        st.info("Re-run `wch run --config ...` to compute land-cover change.")
        return
    ct = lc["change_types"]
    tr = lc["transitions_km2"]
    sparse_to_built = tr.get("sparse vegetation", {}).get("built-up/bare")
    # Plain numbers, no delta arrows: these are areas, not increases.
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [("Vegetation → built-up/bare", "vegetation -> built-up/bare"),
                                        ("Built-up/bare → vegetation", "built-up/bare -> vegetation"),
                                        ("Water change", "water change")]):
        col.metric(label, f"{ct[key]['km2']:.1f} km²")
        col.caption(f"{ct[key]['pct']:.1f}% of the analysed area")
    if sparse_to_built is not None:
        cols[3].metric("Sparse vegetation → built-up/bare", f"{sparse_to_built:.1f} km²")
        cols[3].caption("weaker signal, part of it is construction")
    st.caption("'Vegetation → built-up/bare' is the strongest candidate for urban expansion. Both it and "
               "'sparse → built-up/bare' also contain fallow fields and river sand, which Stage 4's "
               "classifier learns to separate. 'Built-up/bare → vegetation' is mostly bare fields or "
               "sandbars in the before year that were cropped or grassed in the after year.")

    st.subheader("What became what (km²)")
    st.caption("Rows: land cover before. Columns: land cover after. The diagonal is unchanged land.")
    st.dataframe(pd.DataFrame(lc["transitions_km2"]).T)

    st.subheader("Area by land cover (km²)")
    st.dataframe(pd.DataFrame(lc["class_area_km2"]))

    st.subheader("Consistency check")
    shift = lc.get("median_ndvi_diff_stable_built")
    veg_shift = lc.get("median_ndvi_diff_stable_vegetation")
    if shift is not None:
        st.write(f"Median NDVI change on land that is built-up/bare in both years: **{shift:+.3f}**. "
                 "Roofs and roads don't get greener, so a value far from 0 points to a processing or "
                 "sensor difference between the years rather than real change.")
    if veg_shift is not None:
        st.write(f"Median NDVI change on land that is vegetated in both years: **{veg_shift:+.3f}** "
                 "(crop condition and season differences show up here).")

    s = summary["stats"]
    st.subheader("NDVI change (Stage 1-2 view)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean NDVI", f"{s['mean_ndvi_after']:.3f}",
              f"{s['mean_ndvi_after'] - s['mean_ndvi_before']:+.3f} vs before")
    c2.metric("NDVI drop > threshold", f"{s['vegetation_loss_km2']:.1f} km²", delta_color="off")
    c3.metric("NDVI rise > threshold", f"{s['vegetation_gain_km2']:.1f} km²", delta_color="off")

    st.subheader("Data used")
    st.dataframe(scenes_table(summary), hide_index=True)


def tab_timeseries(run_dir: Path, summary: dict) -> None:
    ts_dir = run_dir / "timeseries"
    if not (ts_dir / "monthly.csv").exists():
        st.info("No time series yet. Run `wch timeseries --config config/kanpur.yaml` "
                "(after `wch run`), then reload this page.")
        return
    monthly = pd.read_csv(ts_dir / "monthly.csv")
    ts_summary = json.loads((ts_dir / "timeseries_summary.json").read_text())
    zones = list(monthly["zone"].unique())
    chosen = st.multiselect("Zones", zones, default=zones)

    df = monthly[monthly["zone"].isin(chosen)].copy()
    df["month_start"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    wide = df.pivot_table(index="month_start", columns="zone", values="ndvi")
    st.subheader("Monthly NDVI per zone")
    st.line_chart(wide, height=360)
    st.caption("Each point is the median of up to 2 cloud-free dates in that month (gaps = too cloudy, "
               "usually the monsoon). The yearly peaks are the winter wheat crop. In the "
               "'vegetation → built-up/bare' zone, the year the peaks disappear is roughly when "
               "construction happened.")

    months = ts_summary.get("season_months", [])
    season_path = ts_dir / "season_by_year.csv"
    if season_path.exists():
        st.subheader(f"Mean NDVI in the analysis months ({', '.join(str(m) for m in months)}) by year")
        st.dataframe(pd.read_csv(season_path, index_col=0))
        st.caption("Compare the before and after years with the others: if one of them is unusually "
                   "high or low for stable vegetation, crop-year differences affect the change results.")

    st.subheader("Unusual months")
    anomalies = pd.DataFrame(ts_summary.get("anomalies", []))
    if anomalies.empty:
        st.write("No month stands out in the stable zones.")
    else:
        st.dataframe(anomalies, hide_index=True)
    st.caption("Flagged when a month's NDVI is far from the same month in other years (robust z-score, "
               "only in zones whose land cover did not change).")


# --- Label tab ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def chip(path: str, row: int, col: int, mtime: float) -> np.ndarray:
    """480 m window around a point, enlarged, with the point's 10 m pixel outlined in yellow."""
    half = CHIP_PX // 2
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3], window=Window(col - half, row - half, CHIP_PX, CHIP_PX),
                       boundless=True, fill_value=0)
    z = CHIP_ZOOM
    img = np.repeat(np.repeat(arr.transpose(1, 2, 0), z, axis=0), z, axis=1).copy()
    lo, hi = half * z - 2, (half + 1) * z + 1          # 2 px outline just outside the centre pixel
    yellow = np.array([255, 230, 0], dtype="uint8")
    img[lo:lo + 2, lo:hi + 1] = yellow
    img[hi - 1:hi + 1, lo:hi + 1] = yellow
    img[lo:hi + 1, lo:lo + 2] = yellow
    img[lo:hi + 1, hi - 1:hi + 1] = yellow
    return img


def label_map(point, epsg: int) -> folium.Map:
    """High-res imagery around the point with its 10 m pixel drawn as a square."""
    lons, lats = warp_transform(f"EPSG:{epsg}", "EPSG:4326",
                                [point["x"] - 5, point["x"] + 5], [point["y"] - 5, point["y"] + 5])
    m = folium.Map(location=[point["lat"], point["lon"]], zoom_start=18, max_zoom=19, tiles=None)
    folium.TileLayer(ESRI_URL, attr="Esri World Imagery", max_zoom=19).add_to(m)
    folium.Rectangle([[min(lats), min(lons)], [max(lats), max(lons)]], color="#ffe600", weight=3,
                     fill=False).add_to(m)
    folium.Circle([point["lat"], point["lon"]], radius=40, color="#ffe600", weight=1,
                  dash_array="4", fill=False).add_to(m)
    return m


LABEL_HELP = """
Decide for the **pixel inside the yellow square** (10 m × 10 m):

- **New construction**: it is now a building, road, paved area or active construction site,
  **and** it was field, vegetation or open land in the *before* image.
- **No construction**: anything else, including city that already existed before, fields
  (green, harvested or fallow), trees, river, sand and water.
- **Unsure**: an edge or mixed pixel, or you can't tell. Unsure points are left out of
  training, so use this rather than guess.

Use the *before* chip for what was there then, and the high-res imagery for what is there
now (its capture date is unknown, usually within the last few years). You are not shown
why a point was chosen, so your answer isn't influenced by the rule-based map.
"""


def tab_label(run_dir: Path, summary: dict) -> None:
    pts_path = points_path(LABELS_DIR, run_dir.name)
    lab_path = labels_path(LABELS_DIR, run_dir.name)
    if not pts_path.exists():
        st.info("No points to label yet. Run `wch sample --config config/kanpur.yaml`, then reload.")
        return
    points = pd.read_csv(pts_path)
    labels = read_labels(lab_path)
    labels = labels[labels["id"].isin(points["id"])]
    n_done, n_total = len(labels), len(points)
    st.progress(n_done / n_total, text=f"{n_done} of {n_total} points labelled")
    counts = labels["label"].astype(int).value_counts()
    st.caption(f"New construction: {counts.get(1, 0)} · No construction: {counts.get(0, 0)} · "
               f"Unsure: {counts.get(-1, 0)}. Answers are saved to `{lab_path}` after every click.")

    remaining = points[~points["id"].isin(labels["id"])]
    if remaining.empty:
        st.success("All points labelled. Next: `wch train --config config/kanpur.yaml`.")
    else:
        p = remaining.iloc[0]
        with st.expander("How to label", expanded=n_done == 0):
            st.markdown(LABEL_HELP)
        periods = summary["periods"]
        c1, c2, c3 = st.columns([1, 1, 1.3])
        for col, name in ((c1, "before"), (c2, "after")):
            path = run_dir / f"{name}_rgb.tif"
            col.image(chip(str(path), int(p["row"]), int(p["col"]), path.stat().st_mtime),
                      caption=f"{name.title()}: Sentinel-2 {periods[name]['window'][:7]}, 480 m wide")
        with c3:
            st_folium(label_map(p, summary["epsg"]), height=300, width=None,
                      key=f"labmap_{p['id']}", returned_objects=[])
            st.caption(f"Point {p['id']} · current high-res imagery")
        b1, b2, b3 = st.columns(3)
        answer = None
        if b1.button("New construction", key="lab_yes", type="primary"):
            answer = 1
        if b2.button("No construction", key="lab_no"):
            answer = 0
        if b3.button("Unsure", key="lab_unsure"):
            answer = -1
        if answer is not None:
            append_label(lab_path, p["id"], answer)
            st.rerun()
    if st.button("Undo last answer", key="lab_undo", disabled=labels.empty):
        undo_last_label(lab_path)
        st.rerun()

# --- Model tab ----------------------------------------------------------------------------

def tab_model(run_dir: Path) -> None:
    path = run_dir / "model" / "metrics.json"
    if not path.exists():
        st.info("No model yet. Label points in the Label tab, then run "
                "`wch train --config config/kanpur.yaml` and reload.")
        return
    m = json.loads(path.read_text())
    st.caption(f"Task: {m['task']}. Trained on {m['n_labels']} labels ({m['n_construction']} construction; "
               f"{m['n_unsure_skipped']} unsure left out) from {m['n_spatial_blocks']} spatial blocks.")

    st.subheader("How well does each method find new construction?")
    st.dataframe(pd.DataFrame(m["methods"])[["method", "evaluation", "precision", "recall", "f1", "n"]],
                 hide_index=True)
    st.caption("Precision: of the pixels a method calls construction, the share that really are. "
               "Recall: of the real construction, the share it finds. F1 combines both. "
               "The random forest is always scored on points it did not train on; 'spatial CV' keeps "
               "whole blocks out of training, the random split shows how optimistic ignoring space is. "
               "Points were oversampled where change is likely, so scores describe the labelled set, "
               "not a random pixel.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Random forest mistakes (spatial CV)")
        st.dataframe(pd.DataFrame(m["confusion_matrix_rf_spatial"],
                                  index=["truly no construction", "truly construction"],
                                  columns=["predicted no", "predicted yes"]))
    with c2:
        st.subheader("Which features the model uses")
        st.bar_chart(pd.Series(m["feature_importance"], name="importance"), height=280)

    a = m["area"]
    est = a["stratified_estimate_from_labels"]
    st.subheader("How much new construction?")
    cols = st.columns(4)
    if est["estimate_km2"] is not None:
        cols[0].metric("From your labels", f"{est['estimate_km2']:.1f} km²")
        cols[0].caption(f"95% interval {est['ci95_km2'][0]:.1f}–{est['ci95_km2'][1]:.1f} km²")
    else:
        cols[0].metric("From your labels", "n/a")
        cols[0].caption(est.get("note") or "")
    cols[1].metric("Random forest map", f"{a['random_forest_map_km2']:.1f} km²")
    cols[2].metric("Veg → built-up rule", f"{a['rule_veg_to_built_km2']:.1f} km²")
    cols[3].metric("NDVI drop map", f"{a['ndvi_drop_km2']:.1f} km²")
    st.caption("The label-based figure is the most trustworthy: for each stratum it multiplies the "
               "stratum's area by the share of its labelled points that are construction. The other "
               "three count pixels on a map, so they inherit that method's false alarms and misses.")
    st.dataframe(pd.DataFrame(est["per_stratum"]).T)


def main() -> None:
    st.set_page_config(page_title="What Changed Here?", layout="wide")
    st.title("What Changed Here?")

    runs = sorted(p.parent for p in DATA_DIR.glob("*/summary.json"))
    if not runs:
        st.info(
            f"No results in `{DATA_DIR}` yet. Run `wch run --config config/kanpur.yaml` "
            "from the project root, then reload this page."
        )
        st.stop()

    run_dir = st.sidebar.selectbox("Area", runs, format_func=lambda p: p.name)
    summary = json.loads((run_dir / "summary.json").read_text())
    before, after = summary["periods"]["before"], summary["periods"]["after"]
    st.caption(f"{before['window']} vs {after['window']}; seasonal centres "
               f"{summary.get('seasonal_gap_days', '?')} days apart; {summary['resolution_m']} m pixels.")

    t1, t2, t3, t4, t5 = st.tabs(["Map", "What changed", "Over time", "Label", "Model"])
    with t1:
        tab_map(run_dir, summary)
    with t2:
        tab_changes(summary)
    with t3:
        tab_timeseries(run_dir, summary)
    with t4:
        tab_label(run_dir, summary)
    with t5:
        tab_model(run_dir)


main()