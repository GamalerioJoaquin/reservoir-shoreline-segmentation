"""Masked pixelwise losses used by baseline and shoreline-weighted training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as functional


def _validate_shapes(logits: Tensor, targets: Tensor, valid_mask: Tensor) -> None:
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("logits must have shape (batch, 2, height, width)")
    expected_shape = logits.shape[0], logits.shape[2], logits.shape[3]
    if tuple(targets.shape) != expected_shape or tuple(valid_mask.shape) != expected_shape:
        raise ValueError("targets and valid_mask must match the spatial logit shape")


def edge_weights(distance_squared: Tensor, *, alpha: float, sigma: float) -> Tensor:
    """Compute ``1 + alpha * exp(-d² / (2 * sigma²))`` without caching weights."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return 1.0 + alpha * torch.exp(-distance_squared / (2.0 * sigma**2))


def masked_cross_entropy(logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor | None:
    """Return valid-pixel mean CE, or ``None`` for an entirely invalid batch."""

    _validate_shapes(logits, targets, valid_mask)
    valid = valid_mask.to(dtype=torch.bool)
    valid_count = valid.sum()
    if valid_count.item() == 0:
        return None
    per_pixel = functional.cross_entropy(logits, targets.long(), reduction="none")
    return (per_pixel * valid).sum() / valid_count


def masked_weighted_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    distance_squared: Tensor,
    *,
    alpha: float,
    sigma: float,
) -> Tensor | None:
    """Return the specified valid-pixel weighted CE, without weight normalization.

    The denominator is the count of valid pixels, never the sum of weights.
    """

    _validate_shapes(logits, targets, valid_mask)
    if tuple(distance_squared.shape) != tuple(targets.shape):
        raise ValueError("distance_squared must match targets")
    valid = valid_mask.to(dtype=torch.bool)
    valid_count = valid.sum()
    if valid_count.item() == 0:
        return None
    per_pixel = functional.cross_entropy(logits, targets.long(), reduction="none")
    weights = edge_weights(distance_squared, alpha=alpha, sigma=sigma)
    return (per_pixel * weights * valid).sum() / valid_count
