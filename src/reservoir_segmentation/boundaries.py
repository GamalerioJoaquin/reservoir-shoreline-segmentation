"""Boundary and distance-map utilities for exploratory visualization."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries


def find_boundary_pixels(
    mask: np.ndarray,
    foreground_values: set[int] | tuple[int, ...] | None = None,
    *,
    connectivity: int = 8,
    mode: str = "thick",
) -> np.ndarray:
    """Return a configurable boundary map in a 2-D categorical mask.

    The default is the same thick, diagonal-aware convention used by the
    RiverScope geometry pipeline.  This general visualisation helper has no
    validity-mask input; production distance maps should use
    :func:`reservoir_segmentation.geometry.derive_riverscope_geometry`.
    """

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {array.shape}")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    labels = array if foreground_values is None else np.isin(array, list(foreground_values))
    return find_boundaries(
        labels,
        connectivity=1 if connectivity == 4 else 2,
        mode=mode,
    )


def distance_squared_to_boundary(boundary: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distance to the closest boundary pixel."""

    boundary_array = np.asarray(boundary, dtype=bool)
    if boundary_array.ndim != 2:
        raise ValueError("Boundary map must be 2-D")
    if not boundary_array.any():
        return np.full(boundary_array.shape, np.inf, dtype=float)
    return ndimage.distance_transform_edt(~boundary_array) ** 2


def weights_from_distance_squared(
    distance_squared: np.ndarray,
    sigma: float,
    *,
    base_weight: float = 1.0,
    peak_weight: float = 1.0,
) -> np.ndarray:
    """Convert squared distances into a Gaussian boundary-emphasis map."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    distances = np.asarray(distance_squared, dtype=float)
    return base_weight + peak_weight * np.exp(-distances / (2.0 * sigma**2))
