"""Reusable encoder and decoder blocks for convolutional segmentation models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DoubleConv(nn.Sequential):
    """Apply two 3x3 padded convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    """Extract skip features and reduce their spatial resolution by a factor of two."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return same-resolution skip features and their downsampled representation."""

        skip_features = self.conv(features)
        return skip_features, self.pool(skip_features)


class DecoderBlock(nn.Module):
    """Upsample features, merge a skip connection, and refine the result."""

    def __init__(self, in_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            skip_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(skip_channels * 2, skip_channels)

    def forward(self, features: Tensor, skip_features: Tensor) -> Tensor:
        """Return decoded features at the skip connection's spatial resolution."""

        upsampled_features = self.upconv(features)
        if upsampled_features.shape[-2:] != skip_features.shape[-2:]:
            raise ValueError(
                "Upsampled and skip features must have equal spatial dimensions, "
                f"got {tuple(upsampled_features.shape[-2:])} and "
                f"{tuple(skip_features.shape[-2:])}"
            )
        return self.conv(torch.cat((skip_features, upsampled_features), dim=1))
