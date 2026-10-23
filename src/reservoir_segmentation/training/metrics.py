"""Masked regional and shoreline metrics with explicit degenerate-case records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from reservoir_segmentation.geometry import shoreline_boundary


def _arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(prediction, dtype=bool)
    expected = np.asarray(target, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != expected.shape or expected.shape != valid.shape:
        raise ValueError("prediction, target, and valid_mask must have identical shapes")
    if predicted.ndim != 2:
        raise ValueError("tile metrics expect two-dimensional arrays")
    return predicted, expected, valid


def regional_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Return Dice, IoU, and pixel accuracy over valid pixels only."""

    predicted, expected, valid = _arrays(prediction, target, valid_mask)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {
            "dice": None,
            "iou": None,
            "masked_accuracy": None,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
            "valid_pixels": 0,
        }
    true_positive = int(np.count_nonzero(valid & predicted & expected))
    false_positive = int(np.count_nonzero(valid & predicted & ~expected))
    false_negative = int(np.count_nonzero(valid & ~predicted & expected))
    true_negative = int(np.count_nonzero(valid & ~predicted & ~expected))
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    return {
        "dice": 1.0 if dice_denominator == 0 else 2 * true_positive / dice_denominator,
        "iou": 1.0 if iou_denominator == 0 else true_positive / iou_denominator,
        "masked_accuracy": (true_positive + true_negative) / valid_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "valid_pixels": valid_count,
    }


def merge_regional_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate regional metrics globally by summing valid-pixel confusion counts."""

    fields = ("true_positive", "false_positive", "false_negative", "true_negative")
    totals = {field: sum(int(record[field]) for record in records) for field in fields}
    valid_pixels = sum(totals.values())
    if valid_pixels == 0:
        return regional_metrics(
            np.zeros((1, 1), dtype=bool),
            np.zeros((1, 1), dtype=bool),
            np.zeros((1, 1), dtype=bool),
        )
    true_positive = totals["true_positive"]
    false_positive = totals["false_positive"]
    false_negative = totals["false_negative"]
    true_negative = totals["true_negative"]
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    return {
        "dice": 1.0 if dice_denominator == 0 else 2 * true_positive / dice_denominator,
        "iou": 1.0 if iou_denominator == 0 else true_positive / iou_denominator,
        "masked_accuracy": (true_positive + true_negative) / valid_pixels,
        **totals,
        "valid_pixels": valid_pixels,
    }


def _selection_penalty(shape: tuple[int, int]) -> float:
    return float(np.hypot(*shape))


def boundary_metrics(
    predicted_boundary: np.ndarray,
    reference_boundary: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tolerance_pixels: float = 2.0,
) -> dict[str, float | str | int | None]:
    """Return Boundary F1, ASSD, and HD95 with documented empty cases.

    If only one boundary is empty, standard ASSD and HD95 are undefined and
    recorded as ``None``.  ``selection_assd_px`` uses the tile diagonal as a
    finite, explicit penalty so failed detections cannot vanish from the
    validation objective or geographic aggregation.
    """

    predicted = np.asarray(predicted_boundary, dtype=bool)
    reference = np.asarray(reference_boundary, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.shape != reference.shape or reference.shape != valid.shape:
        raise ValueError("boundary arrays and valid_mask must have identical shapes")
    if predicted.ndim != 2:
        raise ValueError("boundary metrics expect two-dimensional arrays")
    if tolerance_pixels < 0:
        raise ValueError("tolerance_pixels must be non-negative")
    predicted = predicted & valid
    reference = reference & valid
    predicted_count = int(predicted.sum())
    reference_count = int(reference.sum())
    penalty = _selection_penalty(predicted.shape)
    if predicted_count == 0 and reference_count == 0:
        return {
            "boundary_f1": 1.0,
            "assd_px": 0.0,
            "hd95_px": 0.0,
            "selection_assd_px": 0.0,
            "boundary_case": "both_empty",
            "predicted_boundary_pixels": 0,
            "reference_boundary_pixels": 0,
        }
    if predicted_count == 0:
        return {
            "boundary_f1": 0.0,
            "assd_px": None,
            "hd95_px": None,
            "selection_assd_px": penalty,
            "boundary_case": "reference_only",
            "predicted_boundary_pixels": 0,
            "reference_boundary_pixels": reference_count,
        }
    if reference_count == 0:
        return {
            "boundary_f1": 0.0,
            "assd_px": None,
            "hd95_px": None,
            "selection_assd_px": penalty,
            "boundary_case": "prediction_only",
            "predicted_boundary_pixels": predicted_count,
            "reference_boundary_pixels": 0,
        }

    distance_to_reference = ndimage.distance_transform_edt(~reference)
    distance_to_prediction = ndimage.distance_transform_edt(~predicted)
    predicted_distances = distance_to_reference[predicted]
    reference_distances = distance_to_prediction[reference]
    all_distances = np.concatenate((predicted_distances, reference_distances))
    precision = float(np.mean(predicted_distances <= tolerance_pixels))
    recall = float(np.mean(reference_distances <= tolerance_pixels))
    boundary_f1 = (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    assd = float(np.mean(all_distances))
    return {
        "boundary_f1": boundary_f1,
        "assd_px": assd,
        "hd95_px": float(np.percentile(all_distances, 95)),
        "selection_assd_px": assd,
        "boundary_case": "both_present",
        "predicted_boundary_pixels": predicted_count,
        "reference_boundary_pixels": reference_count,
    }


def tile_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    reference_boundary: np.ndarray | None = None,
    tolerance_pixels: float = 2.0,
) -> dict[str, float | str | int | None]:
    """Combine regional and shoreline metrics for one tile."""

    predicted, expected, valid = _arrays(prediction, target, valid_mask)
    if reference_boundary is None:
        reference_boundary = shoreline_boundary(expected, valid, connectivity=8)
    predicted_boundary = shoreline_boundary(predicted, valid, connectivity=8)
    return {
        **regional_metrics(predicted, expected, valid),
        **boundary_metrics(
            predicted_boundary,
            reference_boundary,
            valid,
            tolerance_pixels=tolerance_pixels,
        ),
    }


def aggregate_shoreline_by_group(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate tile shoreline results by geographic group and report failures.

    ``selection_assd_macro_px`` is a macro mean over geographic groups and is
    the validation selection metric.  Standard ASSD and HD95 means include
    only defined values, while the counts of undefined one-empty cases remain
    explicit in the returned summary.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["geographic_group"])].append(record)
    if not grouped:
        return {
            "selection_assd_macro_px": None,
            "boundary_f1_macro": None,
            "assd_defined_mean_px": None,
            "hd95_defined_mean_px": None,
            "group_count": 0,
            "tile_count": 0,
            "degenerate_case_counts": {},
            "groups": {},
        }

    groups: dict[str, dict[str, float | int | None]] = {}
    case_counts: dict[str, int] = defaultdict(int)
    for group, group_records in sorted(grouped.items()):
        for record in group_records:
            case_counts[str(record["boundary_case"])] += 1
        defined_assd = [
            float(record["assd_px"])
            for record in group_records
            if record["assd_px"] is not None
        ]
        defined_hd95 = [
            float(record["hd95_px"])
            for record in group_records
            if record["hd95_px"] is not None
        ]
        groups[group] = {
            "tile_count": len(group_records),
            "selection_assd_px": float(
                np.mean([float(record["selection_assd_px"]) for record in group_records])
            ),
            "boundary_f1": float(
                np.mean([float(record["boundary_f1"]) for record in group_records])
            ),
            "assd_defined_mean_px": float(np.mean(defined_assd)) if defined_assd else None,
            "hd95_defined_mean_px": float(np.mean(defined_hd95)) if defined_hd95 else None,
            "undefined_distance_tiles": sum(record["assd_px"] is None for record in group_records),
        }
    group_values = list(groups.values())
    all_defined_assd = [
        float(record["assd_px"])
        for records in grouped.values()
        for record in records
        if record["assd_px"] is not None
    ]
    all_defined_hd95 = [
        float(record["hd95_px"])
        for records in grouped.values()
        for record in records
        if record["hd95_px"] is not None
    ]
    return {
        "selection_assd_macro_px": float(
            np.mean([float(value["selection_assd_px"]) for value in group_values])
        ),
        "boundary_f1_macro": float(
            np.mean([float(value["boundary_f1"]) for value in group_values])
        ),
        "assd_defined_mean_px": float(np.mean(all_defined_assd)) if all_defined_assd else None,
        "hd95_defined_mean_px": float(np.mean(all_defined_hd95)) if all_defined_hd95 else None,
        "group_count": len(groups),
        "tile_count": sum(len(records) for records in grouped.values()),
        "degenerate_case_counts": dict(sorted(case_counts.items())),
        "groups": groups,
    }
