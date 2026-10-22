"""Small visualization helpers that keep notebook code readable."""

from __future__ import annotations

import numpy as np


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Convert common grayscale/RGB arrays to float RGB in the [0, 1] range."""

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"Expected a grayscale or RGB image, got shape {array.shape}")
    rgb = array[..., :3].astype(float)
    low, high = np.nanpercentile(rgb, [2, 98])
    if high > low:
        rgb = (rgb - low) / (high - low)
    return np.clip(rgb, 0.0, 1.0)


def crop_to_overlap(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to their common top-left spatial extent."""

    if image.ndim < 2 or mask.ndim < 2:
        raise ValueError("Both arrays need at least two dimensions")
    height = min(image.shape[0], mask.shape[0])
    width = min(image.shape[1], mask.shape[1])
    return image[:height, :width, ...], mask[:height, :width]


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[float, float, float] = (1.0, 0.1, 0.1),
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay nonzero mask pixels on an image after spatial overlap cropping."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    rgb = to_display_rgb(image)
    rgb, mask_crop = crop_to_overlap(rgb, np.asarray(mask))
    output = rgb.copy()
    foreground = mask_crop != 0
    output[foreground] = (1.0 - alpha) * output[foreground] + alpha * np.asarray(color)
    return output
