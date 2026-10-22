"""Spatial-padding and prediction helpers for segmentation models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class SpatialPadding:
    """The amount of padding added to each side of a spatial tensor."""

    left: int
    right: int
    top: int
    bottom: int

    @property
    def tuple(self) -> tuple[int, int, int, int]:
        """Return padding in the order required by ``torch.nn.functional.pad``."""

        return (self.left, self.right, self.top, self.bottom)


def pad_to_multiple(
    tensor: Tensor,
    *,
    multiple: int,
    mode: str = "constant",
) -> tuple[Tensor, SpatialPadding]:
    """Symmetrically pad the final two dimensions to the requested multiple."""

    if tensor.ndim < 2:
        raise ValueError("Expected a tensor with at least two spatial dimensions")
    if multiple < 1:
        raise ValueError("multiple must be positive")

    height, width = tensor.shape[-2:]
    total_height_padding = (-height) % multiple
    total_width_padding = (-width) % multiple
    padding = SpatialPadding(
        left=total_width_padding // 2,
        right=total_width_padding - total_width_padding // 2,
        top=total_height_padding // 2,
        bottom=total_height_padding - total_height_padding // 2,
    )
    return functional.pad(tensor, padding.tuple, mode=mode), padding


def unpad(tensor: Tensor, padding: SpatialPadding) -> Tensor:
    """Remove padding from the final two dimensions of a tensor."""

    if tensor.ndim < 2:
        raise ValueError("Expected a tensor with at least two spatial dimensions")
    height_end = -padding.bottom if padding.bottom else None
    width_end = -padding.right if padding.right else None
    return tensor[..., padding.top:height_end, padding.left:width_end]


def probabilities_from_logits(logits: Tensor) -> Tensor:
    """Convert class logits into per-pixel class probabilities."""

    if logits.ndim != 4:
        raise ValueError(
            "Expected logits with shape (batch, classes, height, width), "
            f"got {tuple(logits.shape)}"
        )
    return torch.softmax(logits, dim=1)


def class_labels(probabilities: Tensor) -> Tensor:
    """Return the most probable class index for every predicted pixel."""

    if probabilities.ndim != 4:
        raise ValueError(
            "Expected probabilities with shape (batch, classes, height, width), "
            f"got {tuple(probabilities.shape)}"
        )
    return probabilities.argmax(dim=1)
