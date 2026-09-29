"""Honest evaluation of the construction classifier (Stage 4).

Spatial cross-validation
    Neighbouring pixels look alike. If a random split puts one pixel of a new
    road in training and its neighbour in testing, the model "recognises" the
    answer and the score looks better than it will be on new areas. So points are
    grouped into square blocks (e.g. 2 km) and a whole block is always either
    training or test (GroupKFold). A random split is also reported, to show the gap.

Stratified area estimate
    Labels were drawn with different rates per stratum, so the share of positives
    in the labelled set is not the share in the landscape. The unbiased estimate
    of construction area is  sum over strata of  (stratum area) x (share of its
    labelled points that are construction), with a standard error from the
    binomial variance of each share.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, KFold


def spatial_blocks(x: np.ndarray, y: np.ndarray, block_size: float) -> np.ndarray:
    """Integer block id for each point: which block_size x block_size square it falls in."""
    bx = np.floor(np.asarray(x) / block_size).astype("int64")
    by = np.floor(np.asarray(y) / block_size).astype("int64")
    return bx * 1_000_003 + by


def out_of_fold_proba(
    make_model: Callable, X: np.ndarray, y: np.ndarray,
    groups: np.ndarray | None, n_splits: int, seed: int,
) -> np.ndarray:
    """Probability of class 1 for every point, each predicted by a model that never saw it.

    groups given -> GroupKFold (spatial); groups None -> shuffled KFold (random split).
    """
    proba = np.full(len(y), np.nan)
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
        splits = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    for train, test in splits:
        model = make_model()
        if len(np.unique(y[train])) < 2:   # a fold with one class can't be trained on
            proba[test] = float(y[train][0])
            continue
        model.fit(X[train], y[train])
        proba[test] = model.predict_proba(X[test])[:, 1]
    return proba


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "n": int(len(y_true)),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """[[true neg, false pos], [false neg, true pos]]"""
    return confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
                            labels=[0, 1]).tolist()


def stratified_area_estimate(labelled: pd.DataFrame, strata_area_km2: dict[str, float]) -> dict:
    """Construction area (km2) with a 95% interval, from labels per stratum.

    labelled: DataFrame with columns 'stratum' and 'label' (0/1).
    """
    per_stratum, total, variance, complete = {}, 0.0, 0.0, True
    for name, area in strata_area_km2.items():
        s = labelled.loc[labelled["stratum"] == name, "label"]
        n = len(s)
        if n == 0:
            complete = complete and area == 0   # an empty stratum contributes nothing
            per_stratum[name] = {"area_km2": round(area, 2), "labelled": 0, "positive": 0,
                                 "share": None, "estimate_km2": None}
            continue
        p = float(s.mean())
        total += area * p
        if n > 1:
            variance += area**2 * p * (1 - p) / (n - 1)
        per_stratum[name] = {"area_km2": round(area, 2), "labelled": n, "positive": int(s.sum()),
                             "share": round(p, 3), "estimate_km2": round(area * p, 2)}
    se = float(np.sqrt(variance))
    return {
        "estimate_km2": round(total, 2) if complete else None,
        "ci95_km2": [round(max(total - 1.96 * se, 0.0), 2), round(total + 1.96 * se, 2)] if complete else None,
        "per_stratum": per_stratum,
        "note": None if complete else "some strata have no labels; estimate not possible",
    }