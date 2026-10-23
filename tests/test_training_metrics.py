"""Synthetic checks for masked regional and shoreline metrics."""

import numpy as np

from reservoir_segmentation.training.metrics import (
    aggregate_shoreline_by_group,
    boundary_metrics,
    regional_metrics,
    tile_metrics,
)


def test_regional_metrics_ignore_invalid_pixels():
    target = np.array([[True, False], [False, False]])
    prediction = np.array([[True, True], [True, True]])
    valid = np.array([[True, False], [False, False]])

    metrics = regional_metrics(prediction, target, valid)

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["masked_accuracy"] == 1.0
    assert metrics["valid_pixels"] == 1


def test_boundary_metrics_record_empty_mismatch_without_hiding_it():
    reference = np.zeros((3, 4), dtype=bool)
    reference[1, 1] = True
    predicted = np.zeros_like(reference)
    valid = np.ones_like(reference)

    metrics = boundary_metrics(predicted, reference, valid)

    assert metrics["boundary_case"] == "reference_only"
    assert metrics["boundary_f1"] == 0.0
    assert metrics["assd_px"] is None
    assert metrics["hd95_px"] is None
    assert metrics["selection_assd_px"] == 5.0


def test_both_empty_boundaries_have_zero_distance_and_perfect_boundary_f1():
    empty = np.zeros((2, 2), dtype=bool)

    metrics = boundary_metrics(empty, empty, np.ones_like(empty))

    assert metrics["boundary_case"] == "both_empty"
    assert metrics["boundary_f1"] == 1.0
    assert metrics["assd_px"] == 0.0
    assert metrics["hd95_px"] == 0.0


def test_tile_metrics_do_not_make_invalid_transitions_into_boundaries():
    target = np.array([[False, True], [False, False]])
    prediction = target.copy()
    valid = np.array([[True, False], [True, True]])

    metrics = tile_metrics(prediction, target, valid)

    assert metrics["boundary_case"] == "both_empty"
    assert metrics["boundary_f1"] == 1.0


def test_group_aggregation_keeps_undefined_assd_cases_visible():
    summary = aggregate_shoreline_by_group(
        [
            {
                "geographic_group": "a",
                "boundary_case": "both_present",
                "boundary_f1": 0.5,
                "assd_px": 1.0,
                "hd95_px": 2.0,
                "selection_assd_px": 1.0,
            },
            {
                "geographic_group": "b",
                "boundary_case": "reference_only",
                "boundary_f1": 0.0,
                "assd_px": None,
                "hd95_px": None,
                "selection_assd_px": 5.0,
            },
        ]
    )

    assert summary["selection_assd_macro_px"] == 3.0
    assert summary["degenerate_case_counts"] == {"both_present": 1, "reference_only": 1}
    assert summary["groups"]["b"]["undefined_distance_tiles"] == 1
