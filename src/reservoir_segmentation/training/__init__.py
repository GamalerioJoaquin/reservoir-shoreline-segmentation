"""Reusable training and evaluation components for RiverScope experiments."""

from .config import (
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    SelectionConfig,
)
from .data import (
    ChannelStatistics,
    RiverScopeSegmentationDataset,
    fit_channel_statistics,
    select_overfit_indices,
)
from .engine import evaluate, overfit_smoke_test, run_training, save_training_curves
from .grid_search import make_finalist_repetitions, run_grid_search
from .losses import masked_cross_entropy, masked_weighted_cross_entropy
from .metrics import aggregate_shoreline_by_group, boundary_metrics, regional_metrics
from .reproducibility import set_global_seed

__all__ = [
    "ChannelStatistics",
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RiverScopeSegmentationDataset",
    "RuntimeConfig",
    "SelectionConfig",
    "aggregate_shoreline_by_group",
    "boundary_metrics",
    "evaluate",
    "fit_channel_statistics",
    "make_finalist_repetitions",
    "masked_cross_entropy",
    "masked_weighted_cross_entropy",
    "overfit_smoke_test",
    "regional_metrics",
    "run_grid_search",
    "run_training",
    "save_training_curves",
    "select_overfit_indices",
    "set_global_seed",
]
