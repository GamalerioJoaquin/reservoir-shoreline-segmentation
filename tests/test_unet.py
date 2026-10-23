"""Integration tests for the modular U-Net architecture."""

import torch

from reservoir_segmentation.models import UNet


def test_unet_returns_logits_and_probability_maps_at_original_tile_size():
    model = UNet(in_channels=4, base_channels=2)
    images = torch.zeros((1, 4, 500, 500), requires_grad=True)
    masks = torch.zeros((1, 500, 500), dtype=torch.long)

    logits = model(images)
    probabilities = model.predict_proba(images)

    assert tuple(logits.shape) == (1, 2, 500, 500)
    assert tuple(probabilities.shape) == (1, 2, 500, 500)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones((1, 500, 500)))
    assert torch.nn.CrossEntropyLoss()(logits, masks).ndim == 0
    assert logits.requires_grad
    assert not probabilities.requires_grad


def test_unet_restores_arbitrary_odd_spatial_dimensions():
    model = UNet(in_channels=4, base_channels=2)

    logits = model(torch.randn(1, 4, 37, 35))

    assert tuple(logits.shape) == (1, 2, 37, 35)


def test_unet_backpropagates_through_encoder_and_head():
    model = UNet(in_channels=4, base_channels=2)
    logits = model(torch.randn(1, 4, 32, 32))

    logits.square().mean().backward()

    assert model.head.weight.grad is not None
    assert model.encoders[0].conv[0].weight.grad is not None
