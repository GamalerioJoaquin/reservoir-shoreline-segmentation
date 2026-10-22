"""A modular U-Net implementation for binary reservoir segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import DecoderBlock, DoubleConv, EncoderBlock
from .utils import class_labels, pad_to_multiple, probabilities_from_logits, unpad


class UNet(nn.Module):
    """A configurable U-Net with automatic input padding and output cropping.

    ``forward`` returns unnormalised logits suitable for
    ``torch.nn.CrossEntropyLoss``. ``predict_proba`` returns class
    probabilities for inference without tracking gradients.
    """

    num_classes = 2

    def __init__(
        self,
        in_channels: int = 4,
        *,
        base_channels: int = 32,
        encoder_depth: int = 4,
        padding_mode: str = "constant",
    ) -> None:
        """Create a U-Net with configurable input channels, width, and depth."""

        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be positive")
        if padding_mode not in {"constant", "reflect", "replicate", "circular"}:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")

        self.in_channels = in_channels
        self.padding_mode = padding_mode
        self.padding_multiple = 2**encoder_depth
        feature_channels = [base_channels * 2**level for level in range(encoder_depth)]

        self.encoders = nn.ModuleList()
        channels = in_channels
        for output_channels in feature_channels:
            self.encoders.append(EncoderBlock(channels, output_channels))
            channels = output_channels

        self.bottleneck = DoubleConv(feature_channels[-1], feature_channels[-1] * 2)

        self.decoders = nn.ModuleList()
        channels = feature_channels[-1] * 2
        for skip_channels in reversed(feature_channels):
            self.decoders.append(DecoderBlock(channels, skip_channels))
            channels = skip_channels

        self.head = nn.Conv2d(feature_channels[0], self.num_classes, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        """Return two unnormalised class-score maps matching the input size."""

        self._validate_input(images)
        padded_images, padding = pad_to_multiple(
            images,
            multiple=self.padding_multiple,
            mode=self.padding_mode,
        )

        skip_connections: list[Tensor] = []
        features = padded_images
        for encoder in self.encoders:
            skip_features, features = encoder(features)
            skip_connections.append(skip_features)

        features = self.bottleneck(features)
        for decoder, skip_features in zip(
            self.decoders,
            reversed(skip_connections),
            strict=True,
        ):
            features = decoder(features, skip_features)

        return unpad(self.head(features), padding)

    def forward_logits(self, images: Tensor) -> Tensor:
        """Return logits; retained as an explicit alias for :meth:`forward`."""

        return self.forward(images)

    @torch.no_grad()
    def predict_proba(self, images: Tensor) -> Tensor:
        """Return per-pixel class probabilities without tracking gradients."""

        return probabilities_from_logits(self.forward(images))

    @torch.no_grad()
    def predict_labels(self, images: Tensor) -> Tensor:
        """Return the most likely class label for every input pixel."""

        return class_labels(self.predict_proba(images))

    def _validate_input(self, images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape (batch, channels, height, width), "
                f"got {tuple(images.shape)}"
            )
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {images.shape[1]}"
            )
