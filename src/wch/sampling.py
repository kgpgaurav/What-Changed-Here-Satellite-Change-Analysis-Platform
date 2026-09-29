"""Choose which pixels to label (Stage 4).

Construction covers ~1% of the area, so random points would be almost all
"no construction" and the model would see very few positive examples. We use
**stratified sampling**: split every pixel into one of four strata, then draw a
fixed share of the points from each.

    veg_to_built      vegetation -> built-up/bare      (rule's candidate change)
    sparse_to_built   sparse vegetation -> built-up    (weaker candidate)
    other_change      built -> veg, water change, other sparse changes
    stable            no class change

Oversampling the candidate strata means the labelled set is NOT representative
of the whole area. That is fine for training, but area estimates must re-weight
each stratum by its true size (see evaluate.stratified_area_estimate).

Points are kept at least `min_spacing_m` apart: neighbouring pixels are almost
identical, so close points would add little new information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from wch.landcover import BUILT, NODATA, SPARSE

STRATA = {1: "veg_to_built", 2: "sparse_to_built", 3: "other_change", 4: "stable"}
STRATUM_LABELS = {
    "veg_to_built": "vegetation → built-up/bare",
    "sparse_to_built": "sparse vegetation → built-up/bare",
    "other_change": "other change",
    "stable": "no class change",
}


def strata_map(before_lc: xr.DataArray, after_lc: xr.DataArray, changes: xr.DataArray) -> xr.DataArray:
    """uint8 map of stratum codes (keys of STRATA), 0 = no data. Every valid pixel gets one."""
    b, a, c = before_lc, after_lc, changes
    sparse_to_built = (b == SPARSE) & (a == BUILT)
    out = xr.where(c.isin([1, 2, 3]) | ((c == 7) & (b == a)), 4, 3)   # stable, else other change
    out = xr.where(sparse_to_built, 2, out)
    out = xr.where(c == 4, 1, out)
    out = xr.where((b == NODATA) | (a == NODATA) | (c == 0), 0, out)
    return out.astype("uint8").rename("stratum")


def allocate(n_points: int, shares: dict[str, float]) -> dict[str, int]:
    """Split n_points over strata by share; rounding leftovers go to the largest shares."""
    raw = {k: n_points * v / sum(shares.values()) for k, v in shares.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n_points - sum(counts.values())]:
        counts[k] += 1
    return counts


def sample_points(
    strata: np.ndarray, x: np.ndarray, y: np.ndarray,
    counts: dict[str, int], min_spacing: float, rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw spaced random pixels per stratum.

    strata: 2-D stratum codes; x, y: 1-D map coordinates of columns / rows (metres).
    Returns rows (stratum, row, col, x, y), shuffled so strata are mixed.
    """
    accepted_xy = np.empty((0, 2))
    records = []
    for code, name in STRATA.items():
        need = counts.get(name, 0)
        if need == 0:
            continue
        flat = np.flatnonzero(strata == code)
        # Look at a random subset of candidates; far more than needed so spacing can be met.
        candidates = rng.permutation(flat)[: max(need * 50, 1000)]
        taken = 0
        for idx in candidates:
            if taken == need:
                break
            r, c = divmod(int(idx), strata.shape[1])
            p = np.array([x[c], y[r]])
            if len(accepted_xy) and np.min(np.hypot(*(accepted_xy - p).T)) < min_spacing:
                continue
            accepted_xy = np.vstack([accepted_xy, p])
            records.append({"stratum": name, "row": r, "col": c, "x": float(x[c]), "y": float(y[r])})
            taken += 1
    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)