"""Unit tests for reusable U-Net building blocks and spatial helpers."""

import torch

from reservoir_segmentation.models.blocks import DecoderBlock, EncoderBlock
from reservoir_segmentation.models.utils import pad_to_multiple, unpad


def test_encoder_block_returns_skip_and_downsampled_features():
    encoder = EncoderBlock(in_channels=4, out_channels=8)

    skip_features, downsampled_features = encoder(torch.randn(2, 4, 64, 64))

    assert tuple(skip_features.shape) == (2, 8, 64, 64)
    assert tuple(downsampled_features.shape) == (2, 8, 32, 32)


def test_decoder_block_upsamples_and_merges_skip_features():
    decoder = DecoderBlock(in_channels=128, skip_channels=64)
    features = torch.randn(2, 128, 32, 32)
    skip_features = torch.randn(2, 64, 64, 64)

    output = decoder(features, skip_features)

    assert tuple(output.shape) == (2, 64, 64, 64)


def test_padding_round_trip_preserves_a_500_pixel_tile():
    images = torch.randn(1, 4, 500, 500)

    padded_images, padding = pad_to_multiple(images, multiple=16)

    assert tuple(padded_images.shape) == (1, 4, 512, 512)
    assert torch.equal(unpad(padded_images, padding), images)
