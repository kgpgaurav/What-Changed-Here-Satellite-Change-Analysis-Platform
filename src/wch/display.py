"""Turn analysis arrays into images people can look at.

These functions only affect visuals; all statistics use the real float values.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps


def stretch_rgb(red, green, blue, max_reflectance: float = 0.3) -> np.ndarray:
    """Reflectance -> 8-bit true-colour image of shape (3, H, W).

    Land reflectance rarely exceeds ~0.3, so mapping 0-0.3 onto 1-255 gives a
    natural-looking picture. 0 is reserved as 'no data'.
    """
    bands = []
    for band in (red, green, blue):
        arr = np.asarray(band, dtype="float32")
        scaled = 1 + np.clip(arr / max_reflectance, 0, 1) * 254
        bands.append(np.where(np.isnan(arr), 0, scaled).astype("uint8"))
    return np.stack(bands)


def colorize(arr: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    """Single-band values -> RGBA image (H, W, 4); NaN pixels become transparent."""
    arr = np.asarray(arr, dtype="float32")
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(norm)) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isnan(arr), 0, 255)
    return rgba


def colorize_classes(arr: np.ndarray, palette: dict[int, str]) -> np.ndarray:
    """Class codes -> RGBA image (H, W, 4). Codes not in the palette (e.g. 0 = no data)
    are transparent. palette maps code -> hex colour like '#1a9850'."""
    arr = np.nan_to_num(np.asarray(arr), nan=0).astype("int64")
    rgba = np.zeros(arr.shape + (4,), dtype="uint8")
    for code, hex_colour in palette.items():
        h = hex_colour.lstrip("#")
        rgba[arr == code] = [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255]
    return rgba