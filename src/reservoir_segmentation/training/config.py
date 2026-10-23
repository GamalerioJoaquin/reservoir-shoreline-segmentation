"""Serializable configuration objects for segmentation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters for the in-repository U-Net."""

    in_channels: int = 4
    base_channels: int = 32
    encoder_depth: int = 4
    padding_mode: str = "constant"


@dataclass(frozen=True)
class LossConfig:
    """Parameters for either baseline or shoreline-weighted cross entropy."""

    name: str = "cross_entropy"
    alpha: float = 0.0
    sigma: float | None = None

    def __post_init__(self) -> None:
        if self.name not in {"cross_entropy", "weighted_cross_entropy"}:
            raise ValueError("name must be cross_entropy or weighted_cross_entropy")
        if self.name == "weighted_cross_entropy":
            if self.alpha < 0:
                raise ValueError("alpha must be non-negative")
            if self.sigma is None or self.sigma <= 0:
                raise ValueError("weighted_cross_entropy requires a positive sigma")


@dataclass(frozen=True)
class OptimizationConfig:
    """AdamW and scheduler parameters."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 100
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    scheduler_min_delta: float = 0.01
    min_learning_rate: float = 1e-7

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.scheduler_patience < 0 or not 0 < self.scheduler_factor < 1:
            raise ValueError("invalid ReduceLROnPlateau configuration")


@dataclass(frozen=True)
class RuntimeConfig:
    """Reproducibility and accelerator configuration."""

    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    deterministic: bool = True
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class SelectionConfig:
    """Validation checkpoint and early-stopping policy.

    ASSD is minimized inside a Dice safety band.  A candidate whose Dice is
    more than ``dice_tolerance`` below the current checkpoint is rejected;
    a materially better Dice can replace the checkpoint even with a worse
    ASSD.  This prevents shoreline gains from concealing regional collapse.
    """

    early_stopping_patience: int = 20
    dice_tolerance: float = 0.01
    assd_min_delta: float = 0.01

    def __post_init__(self) -> None:
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.dice_tolerance < 0 or self.assd_min_delta < 0:
            raise ValueError("selection deltas must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, JSON-serializable description of one training run."""

    label: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        """Rebuild a configuration previously written by :meth:`to_dict`."""

        return cls(
            label=str(values.get("label", "")),
            model=ModelConfig(**values.get("model", {})),
            loss=LossConfig(**values.get("loss", {})),
            optimization=OptimizationConfig(**values.get("optimization", {})),
            runtime=RuntimeConfig(**values.get("runtime", {})),
            selection=SelectionConfig(**values.get("selection", {})),
        )
