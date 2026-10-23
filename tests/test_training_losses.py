"""Synthetic checks for valid-pixel segmentation losses."""

import math

import torch

from reservoir_segmentation.training.losses import (
    masked_cross_entropy,
    masked_weighted_cross_entropy,
)


def test_masked_cross_entropy_excludes_invalid_pixels_completely():
    targets = torch.tensor([[[0, 1], [0, 1]]])
    valid = torch.tensor([[[True, True], [False, False]]])
    logits = torch.tensor(
        [[[[5.0, -5.0], [-100.0, 100.0]], [[-5.0, 5.0], [100.0, -100.0]]]]
    )
    baseline = masked_cross_entropy(logits, targets, valid)
    changed_invalid_logits = logits.clone()
    changed_invalid_logits[:, :, 1, :] *= -1
    changed = masked_cross_entropy(changed_invalid_logits, targets, valid)

    assert baseline is not None and changed is not None
    assert torch.allclose(baseline, changed)


def test_weighted_cross_entropy_divides_by_valid_count_not_sum_of_weights():
    logits = torch.zeros((1, 2, 1, 2))
    targets = torch.tensor([[[0, 1]]])
    valid = torch.tensor([[[True, True]]])
    distance_squared = torch.tensor([[[0.0, float("inf")]]])

    loss = masked_weighted_cross_entropy(
        logits,
        targets,
        valid,
        distance_squared,
        alpha=2.0,
        sigma=1.0,
    )

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(2.0 * math.log(2.0)))


def test_losses_skip_a_batch_with_no_valid_pixels():
    logits = torch.zeros((1, 2, 2, 2))
    targets = torch.zeros((1, 2, 2), dtype=torch.long)
    valid = torch.zeros((1, 2, 2), dtype=torch.bool)
    distance_squared = torch.zeros((1, 2, 2))

    assert masked_cross_entropy(logits, targets, valid) is None
    assert (
        masked_weighted_cross_entropy(
            logits,
            targets,
            valid,
            distance_squared,
            alpha=1.0,
            sigma=2.0,
        )
        is None
    )
