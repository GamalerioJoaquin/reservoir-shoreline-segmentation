"""PyTorch model definitions for reservoir segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class DoubleConvolution(nn.Sequential):
    """Apply two 3x3 padded convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """A padded U-Net for binary water segmentation of 500x500 RiverScope tiles.

    Inputs are padded symmetrically from 500x500 to 512x512 before entering
    the network. The encoder has four max-pooling stages and the decoder uses
    transposed convolutions with skip connections. Output padding is removed,
    so both logits and probabilities retain the physical 500x500 tile extent.

    ``forward`` returns per-class probabilities after softmax. Use
    :meth:`forward_logits` with ``torch.nn.CrossEntropyLoss`` during training.
    """

    input_size = (500, 500)
    padded_size = (512, 512)
    num_classes = 2

    def __init__(
        self,
        in_channels: int = 4,
        *,
        base_channels: int = 32,
        encoder_depth: int = 4,
        padding_mode: str = "constant",
    ) -> None:
        """Create a U-Net with configurable input and feature-channel counts."""

        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if encoder_depth != 4:
            raise ValueError("encoder_depth must be 4 for 512x512 input tiles")
        if padding_mode not in {"constant", "reflect", "replicate", "circular"}:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")

        self.padding_mode = padding_mode
        features = [base_channels * 2**level for level in range(encoder_depth)]

        self.encoder_blocks = nn.ModuleList()
        channels = in_channels
        for feature_channels in features:
            self.encoder_blocks.append(DoubleConvolution(channels, feature_channels))
            channels = feature_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConvolution(features[-1], features[-1] * 2)

        self.upconvolutions = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        decoder_channels = features[-1] * 2
        for skip_channels in reversed(features):
            self.upconvolutions.append(
                nn.ConvTranspose2d(
                    decoder_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoder_blocks.append(DoubleConvolution(skip_channels * 2, skip_channels))
            decoder_channels = skip_channels

        self.output_layer = nn.Conv2d(features[0], self.num_classes, kernel_size=1)

    def forward_logits(self, images: Tensor) -> Tensor:
        """Return two unnormalised 500x500 class-score maps for each image."""

        self._validate_input(images)
        padded_images = self._pad_to_network_size(images)

        skip_connections: list[Tensor] = []
        features = padded_images
        for encoder_block in self.encoder_blocks:
            features = encoder_block(features)
            skip_connections.append(features)
            features = self.pool(features)

        features = self.bottleneck(features)
        for upconvolution, decoder_block, skip_connection in zip(
            self.upconvolutions,
            self.decoder_blocks,
            reversed(skip_connections),
            strict=True,
        ):
            features = upconvolution(features)
            features = decoder_block(torch.cat((skip_connection, features), dim=1))

        padded_logits = self.output_layer(features)
        return self._remove_padding(padded_logits)

    def forward(self, images: Tensor) -> Tensor:
        """Return class-0 and class-1 probabilities for each 500x500 pixel."""

        return torch.softmax(self.forward_logits(images), dim=1)

    def _validate_input(self, images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape (batch, channels, height, width), "
                f"got {tuple(images.shape)}"
            )
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size[0]}x{self.input_size[1]} inputs, "
                f"got {tuple(images.shape[-2:])}"
            )

    def _pad_to_network_size(self, images: Tensor) -> Tensor:
        height_padding = self.padded_size[0] - self.input_size[0]
        width_padding = self.padded_size[1] - self.input_size[1]
        padding = (
            width_padding // 2,
            width_padding - width_padding // 2,
            height_padding // 2,
            height_padding - height_padding // 2,
        )
        return functional.pad(images, padding, mode=self.padding_mode)

    def _remove_padding(self, output: Tensor) -> Tensor:
        height_padding = (self.padded_size[0] - self.input_size[0]) // 2
        width_padding = (self.padded_size[1] - self.input_size[1]) // 2
        return output[
            ...,
            height_padding : height_padding + self.input_size[0],
            width_padding : width_padding + self.input_size[1],
        ]


def class_labels(probabilities: Tensor) -> Tensor:
    """Return the most probable class index for every predicted pixel."""

    if probabilities.ndim != 4 or probabilities.shape[1] != UNet.num_classes:
        raise ValueError(
            "Expected probabilities with shape (batch, 2, height, width), "
            f"got {tuple(probabilities.shape)}"
        )
    return probabilities.argmax(dim=1)


def expected_output_shape(batch_size: int) -> Sequence[int]:
    """Return the fixed probability-map shape emitted for a batch of tiles."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return (batch_size, UNet.num_classes, *UNet.input_size)
