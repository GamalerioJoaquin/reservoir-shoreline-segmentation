import numpy as np
import pytest

from reservoir_segmentation.geometry import (
    binary_water_mask,
    derive_riverscope_geometry,
    shoreline_boundary,
    valid_mask_from_labels,
)


def test_geometry_marks_diagonal_water_land_contacts_with_eight_connectivity():
    labels = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    valid = valid_mask_from_labels(labels)
    water = binary_water_mask(labels, valid)

    four_connected = shoreline_boundary(water, valid, connectivity=4)
    eight_connected = shoreline_boundary(water, valid, connectivity=8)

    assert not four_connected[1, 1]
    assert eight_connected[1, 1]
    assert np.all(eight_connected)


def test_invalid_pixels_and_outer_tile_frame_do_not_create_a_shoreline():
    labels = np.array([[1, 1, 1]], dtype=np.uint8)
    valid = np.array([[True, False, True]])
    water = binary_water_mask(labels, valid)
    boundary = shoreline_boundary(water, valid)

    assert not boundary.any()

    maps = derive_riverscope_geometry(np.ones((3, 3), dtype=np.uint8))
    assert not maps.boundary.any()
    assert np.isinf(maps.distance_squared).all()


def test_geometry_preserves_binary_labels_and_uses_float32_distance_maps():
    labels = np.array(
        [[0, 0, 0], [0, 2, 1], [0, 0, 0]],
        dtype=np.uint8,
    )

    maps = derive_riverscope_geometry(labels)

    assert maps.binary_mask.dtype == bool
    assert maps.valid_mask.all()
    assert maps.binary_mask[1, 1]
    assert maps.binary_mask[1, 2]
    assert maps.boundary.any()
    assert maps.distance_squared.dtype == np.float32
    assert maps.distance_squared[maps.boundary].max() == 0


def test_undocumented_riverscope_label_three_is_invalid_by_default():
    maps = derive_riverscope_geometry(np.array([[0, 3]], dtype=np.uint8))

    assert not maps.valid_mask[0, 1]
    assert not maps.binary_mask[0, 1]
    assert not maps.boundary.any()


def test_unknown_valid_labels_require_an_explicit_policy():
    labels = np.array([[0, 4]], dtype=np.uint8)

    with pytest.raises(ValueError, match="Unexpected valid RiverScope label values"):
        derive_riverscope_geometry(labels)

    maps = derive_riverscope_geometry(labels, nodata_values=(3, 4))
    assert not maps.valid_mask[0, 1]
