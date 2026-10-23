"""Reproducible geometry derived from RiverScope segmentation labels.

The functions in this module deliberately operate on labels rather than on
imagery.  A shoreline is the interface between two valid pixels with
different water classes; it is not an image edge.  This keeps the definition
stable across visualisations, distance maps, losses, and raster metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

RIVERSCOPE_WATER_LABELS = (1, 2)
RIVERSCOPE_IGNORED_LABELS = (3,)
GEOMETRY_VERSION = "riverscope-geometry-v2"


@dataclass(frozen=True)
class GeometryMaps:
    """Derived arrays for one unpadded RiverScope label tile.

    ``boundary`` marks both valid pixels adjacent to a class change.  This is
    a thick, 8-connected raster convention.  Invalid pixels never form a
    boundary and are excluded by ``valid`` downstream.
    """

    binary_mask: np.ndarray
    valid_mask: np.ndarray
    boundary: np.ndarray
    distance_squared: np.ndarray


def _require_2d(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {result.shape}")
    return result


def valid_mask_from_labels(
    labels: np.ndarray,
    *,
    nodata_values: Iterable[float | int] = (),
) -> np.ndarray:
    """Return validity from finite labels and explicitly declared nodata.

    RiverScope does not publish a standalone validity mask.  Consequently,
    this function does *not* infer clouds, shadows, or image nodata.  Those
    decisions must be added explicitly by a later, documented policy.
    """

    array = _require_2d(labels, "labels")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"labels must be numeric, got {array.dtype}")
    valid = np.isfinite(array)
    for value in nodata_values:
        valid &= array != value
    return valid.astype(bool, copy=False)


def binary_water_mask(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    water_labels: Iterable[int] = RIVERSCOPE_WATER_LABELS,
) -> np.ndarray:
    """Convert valid RiverScope labels to a binary water mask.

    The original multiclass label is intentionally not modified or copied.
    Valid values outside ``0`` plus ``water_labels`` are rejected instead of
    silently becoming land or an invalid region.
    """

    array = _require_2d(labels, "labels")
    valid = _require_2d(valid_mask, "valid_mask").astype(bool, copy=False)
    if array.shape != valid.shape:
        raise ValueError("labels and valid_mask must have the same shape")

    water_values = tuple(water_labels)
    allowed_values = (0, *water_values)
    unexpected = np.unique(array[valid & ~np.isin(array, allowed_values)])
    if unexpected.size:
        raise ValueError(
            "Unexpected valid RiverScope label values: "
            f"{unexpected.tolist()}. Declare them as nodata or update the label policy."
        )

    binary = np.isin(array, water_values)
    binary &= valid
    return binary.astype(bool, copy=False)


def shoreline_boundary(
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    connectivity: int = 8,
) -> np.ndarray:
    """Return a thick shoreline map from valid label transitions.

    Adjacent pairs are compared directly, making it impossible for the outer
    image frame or a valid/invalid transition to become an artificial coast.
    With ``connectivity=8`` diagonal neighbours are also compared, avoiding a
    directional preference for horizontal and vertical shorelines.
    """

    water = _require_2d(binary_mask, "binary_mask").astype(bool, copy=False)
    valid = _require_2d(valid_mask, "valid_mask").astype(bool, copy=False)
    if water.shape != valid.shape:
        raise ValueError("binary_mask and valid_mask must have the same shape")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")

    boundary = np.zeros(water.shape, dtype=bool)
    offsets = [(0, 1), (1, 0)]
    if connectivity == 8:
        offsets.extend([(1, 1), (1, -1)])

    height, width = water.shape
    for row_offset, column_offset in offsets:
        row_start = max(0, -row_offset)
        row_stop = height - max(0, row_offset)
        column_start = max(0, -column_offset)
        column_stop = width - max(0, column_offset)
        source = (slice(row_start, row_stop), slice(column_start, column_stop))
        neighbour = (
            slice(row_start + row_offset, row_stop + row_offset),
            slice(column_start + column_offset, column_stop + column_offset),
        )
        transition = (
            valid[source]
            & valid[neighbour]
            & (water[source] != water[neighbour])
        )
        boundary[source] |= transition
        boundary[neighbour] |= transition
    return boundary


def distance_squared_to_shoreline(boundary: np.ndarray) -> np.ndarray:
    """Return float32 squared Euclidean distance to the nearest shoreline.

    A tile with no valid class transition has no physical coastline in this
    representation, so it receives ``inf`` rather than a distance invented at
    the outer edge of the tile.
    """

    shoreline = _require_2d(boundary, "boundary").astype(bool, copy=False)
    if not shoreline.any():
        return np.full(shoreline.shape, np.inf, dtype=np.float32)
    distances = ndimage.distance_transform_edt(~shoreline)
    return np.square(distances, dtype=np.float32)


def derive_riverscope_geometry(
    labels: np.ndarray,
    *,
    nodata_values: Iterable[float | int] = RIVERSCOPE_IGNORED_LABELS,
    connectivity: int = 8,
) -> GeometryMaps:
    """Derive binary labels, validity, boundary, and ``d²`` before padding.

    RiverScope value ``3`` is an observed but undocumented source value. It
    is treated as invalid by default: it contributes neither to the target,
    shoreline, loss, nor metrics. No padding or crop is performed here.
    Calling code must preserve these arrays' alignment if it later pads data
    for a U-Net input size.
    """

    valid = valid_mask_from_labels(labels, nodata_values=nodata_values)
    binary = binary_water_mask(labels, valid)
    boundary = shoreline_boundary(binary, valid, connectivity=connectivity)
    return GeometryMaps(
        binary_mask=binary,
        valid_mask=valid,
        boundary=boundary,
        distance_squared=distance_squared_to_shoreline(boundary),
    )
