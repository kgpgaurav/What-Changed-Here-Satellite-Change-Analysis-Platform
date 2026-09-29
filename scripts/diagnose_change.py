"""Where do NDVI gain and loss come from? Reads existing outputs only (no download).

    python scripts/diagnose_change.py                          # data/processed/kanpur
    python scripts/diagnose_change.py data/processed/<name>

Uses NDVI ranges as a rough land-cover proxy (Stage 3 adds proper built-up/water indices):
    water < 0 <= built-up/bare < 0.25 <= sparse veg < 0.5 <= dense veg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray

CLASSES = ["water", "built-up/bare", "sparse veg", "dense veg"]
EDGES = [0.0, 0.25, 0.5]  # NDVI class boundaries


def classify(ndvi: np.ndarray) -> np.ndarray:
    """0..3 = CLASSES index, -1 = no data."""
    cls = np.digitize(ndvi, EDGES)
    return np.where(np.isnan(ndvi), -1, cls)


def read(path: Path) -> np.ndarray:
    return rioxarray.open_rasterio(path, masked=True).values[0].astype("float64")


def main(run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    km2 = summary["resolution_m"] ** 2 / 1e6
    thr = summary["stats"]["threshold"]
    before, after = read(run_dir / "before_ndvi.tif"), read(run_dir / "after_ndvi.tif")
    diff = after - before
    valid = ~np.isnan(diff)
    d = diff[valid]
    cb, ca = classify(before)[valid], classify(after)[valid]
    pd.set_option("display.width", 120)

    print("\n1) NDVI change distribution (after - before), all valid pixels")
    pct = np.percentile(d, [5, 25, 50, 75, 95])
    print("   percentiles 5/25/50/75/95: " + " / ".join(f"{p:+.3f}" for p in pct))
    print(f"   share nearly unchanged (|diff| < 0.05): {100 * np.mean(np.abs(d) < 0.05):.1f}%")
    print("   -> if the median is clearly above 0, the WHOLE area shifted, not just some places.")

    print("\n2) By land-cover class in the BEFORE year")
    rows = []
    for i, name in enumerate(CLASSES):
        m = cb == i
        if m.any():
            rows.append({"class (before)": name, "area km2": round(m.sum() * km2, 1),
                         "median diff": round(float(np.median(d[m])), 3),
                         "gain km2": round(np.sum(d[m] > thr) * km2, 1),
                         "loss km2": round(np.sum(d[m] < -thr) * km2, 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n3) Transitions: area (km2) by class before (rows) -> after (columns)")
    matrix = pd.DataFrame(0.0, index=CLASSES, columns=CLASSES)
    for i in range(4):
        for j in range(4):
            matrix.iloc[i, j] = round(np.sum((cb == i) & (ca == j)) * km2, 1)
    print(matrix.to_string())
    print("   off-diagonal cells are class changes; 'dense veg -> built-up/bare' contains construction")
    print("   but also fields that were harvested or left fallow.")

    print("\n4) Reference check: pixels that are built-up/bare in BOTH years")
    stable = (cb == 1) & (ca == 1)
    if stable.any():
        print(f"   {stable.sum() * km2:.1f} km2, median diff {np.median(d[stable]):+.3f}")
        print("   -> should be close to 0. A consistent shift here suggests a processing/sensor")
        print("      difference between the years rather than real change.")

    print("\n5) Scenes behind each composite")
    for label, p in summary["periods"].items():
        sats = sorted({s.split("_")[0] for s in p.get("scene_ids", [])})
        rules = sorted(set(p.get("offset_rules", {}).values()))
        print(f"   {label}: {len(p.get('dates', []))} dates, collections {p.get('collections')}, "
              f"satellites {sats}, offset rules {rules}")
    print(f"   seasonal gap: {summary.get('seasonal_gap_days')} days\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/kanpur"))