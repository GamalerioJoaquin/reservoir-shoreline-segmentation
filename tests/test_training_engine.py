"""Synthetic checks for checkpoint selection and split safeguards."""

from reservoir_segmentation.training.config import SelectionConfig
from reservoir_segmentation.training.engine import CheckpointSelector


def test_checkpoint_selector_prioritizes_lower_assd_inside_dice_safety_band():
    selector = CheckpointSelector(SelectionConfig(dice_tolerance=0.01, assd_min_delta=0.01))

    assert selector.consider(epoch=1, dice=0.80, assd=4.0) == "initial"
    assert (
        selector.consider(epoch=2, dice=0.795, assd=3.0)
        == "lower_assd_within_dice_band"
    )
    assert selector.best_epoch == 2
    assert selector.consider(epoch=3, dice=0.78, assd=1.0) is None
    assert selector.best_epoch == 2


def test_checkpoint_selector_allows_material_dice_recovery():
    selector = CheckpointSelector(SelectionConfig(dice_tolerance=0.01, assd_min_delta=0.01))

    selector.consider(epoch=1, dice=0.70, assd=2.0)

    assert selector.consider(epoch=2, dice=0.72, assd=3.0) == "material_dice_improvement"
    assert selector.best_epoch == 2
