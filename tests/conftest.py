import numpy as np
import pytest
import xarray as xr


def make_scene(red, nir, green=None, blue=None) -> xr.Dataset:
    """Tiny synthetic reflectance scene on a fake 10 m UTM grid."""
    red = np.asarray(red, dtype="float32")
    ny, nx = red.shape
    coords = {"y": 3_000_000 - 10 * np.arange(ny) - 5, "x": 400_000 + 10 * np.arange(nx) + 5}
    def band(v):
        return xr.DataArray(np.asarray(v, dtype="float32"), dims=("y", "x"), coords=coords)
    return xr.Dataset({
        "red": band(red),
        "nir": band(nir),
        "green": band(green if green is not None else red),
        "blue": band(blue if blue is not None else red),
    })


@pytest.fixture
def scene_factory():
    return make_scene
