"""Resumable train/validation grid-search orchestration without test access."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import build_dataloader
from .engine import run_training
from .experiments import GridSearchStore, canonical_config_hash


def _existing_trial_record(trial_root: Path) -> dict[str, Any] | None:
    summary_path = trial_root / "summary.json"
    status_path = trial_root / "status.json"
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    record: dict[str, Any] = {
        "config_hash": trial_root.name,
        "state": status.get("state"),
        "trial_path": str(trial_root),
    }
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record.update(
            {
                "best_epoch": summary.get("best_epoch"),
                "validation_dice": summary.get("best_validation_dice"),
                "selection_assd_macro_px": summary.get(
                    "best_validation_selection_assd_macro_px"
                ),
            }
        )
    else:
        record["error"] = status.get("message")
    return record


def _select_winner(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    completed = [
        dict(record)
        for record in records
        if record.get("state") == "completed"
        and record.get("selection_assd_macro_px") is not None
        and record.get("validation_dice") is not None
    ]
    if not completed:
        return None
    return min(
        completed,
        key=lambda record: (
            float(record["selection_assd_macro_px"]),
            -float(record["validation_dice"]),
        ),
    )


def run_grid_search(
    configurations: Iterable[ExperimentConfig],
    train_dataset: Dataset[dict[str, Any]],
    validation_dataset: Dataset[dict[str, Any]],
    *,
    output_dir: str | Path,
    channel_statistics: Mapping[str, Any],
    repository_root: str | Path,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Run pending train/validation trials and persist each terminal outcome.

    Completed configurations are always skipped by their full configuration
    hash.  Failed trials remain visible; they are retried only when explicitly
    requested, avoiding an accidental loop over a persistent environment or
    data error.  This function deliberately has no test-loader parameter.
    """

    store = GridSearchStore(output_dir)
    configurations = list(configurations)
    configurations_by_hash = {
        canonical_config_hash(configuration): configuration for configuration in configurations
    }
    records: dict[str, dict[str, Any]] = {}
    for configuration in configurations:
        trial_root = store.trial_root(configuration)
        existing = _existing_trial_record(trial_root)
        if existing is not None:
            records[canonical_config_hash(configuration)] = existing
    progress = tqdm(configurations, desc="grid-search trials")
    for configuration in progress:
        config_hash = canonical_config_hash(configuration)
        existing = records.get(config_hash)
        if existing is not None and (
            existing.get("state") == "completed"
            or (existing.get("state") == "failed" and not retry_failed)
        ):
            progress.set_postfix(skipped=config_hash)
            continue
        trial_root = store.trial_root(configuration)
        try:
            train_loader = build_dataloader(
                train_dataset,
                batch_size=configuration.optimization.batch_size,
                shuffle=True,
                seed=configuration.runtime.seed,
                num_workers=configuration.runtime.num_workers,
            )
            validation_loader = build_dataloader(
                validation_dataset,
                batch_size=configuration.optimization.batch_size,
                shuffle=False,
                seed=configuration.runtime.seed,
                num_workers=configuration.runtime.num_workers,
            )
            result = run_training(
                train_loader,
                validation_loader,
                configuration,
                output_dir=trial_root,
                channel_statistics=channel_statistics,
                repository_root=repository_root,
            )
            summary = result["summary"]
            records[config_hash] = {
                "config_hash": config_hash,
                "state": "completed",
                "trial_path": str(trial_root),
                "label": configuration.label,
                "best_epoch": summary["best_epoch"],
                "validation_dice": summary["best_validation_dice"],
                "selection_assd_macro_px": summary[
                    "best_validation_selection_assd_macro_px"
                ],
                "loss_name": configuration.loss.name,
                "learning_rate": configuration.optimization.learning_rate,
                "base_channels": configuration.model.base_channels,
                "batch_size": configuration.optimization.batch_size,
                "seed": configuration.runtime.seed,
                "alpha": configuration.loss.alpha,
                "sigma": configuration.loss.sigma,
            }
        except BaseException as error:
            records[config_hash] = {
                "config_hash": config_hash,
                "state": "failed",
                "trial_path": str(trial_root),
                "label": configuration.label,
                "error_type": type(error).__name__,
                "error": str(error),
                "loss_name": configuration.loss.name,
                "learning_rate": configuration.optimization.learning_rate,
                "base_channels": configuration.model.base_channels,
                "batch_size": configuration.optimization.batch_size,
                "seed": configuration.runtime.seed,
                "alpha": configuration.loss.alpha,
                "sigma": configuration.loss.sigma,
            }
        sorted_records = sorted(
            records.values(),
            key=lambda record: (
                record.get("state") != "completed",
                record.get("selection_assd_macro_px") is None,
                record.get("selection_assd_macro_px", float("inf")),
            ),
        )
        store.write_results(sorted_records)
        winner = _select_winner(sorted_records)
        if winner is not None:
            winner["experiment_config"] = configurations_by_hash[winner["config_hash"]].to_dict()
            store.write_winner(winner)
        progress.set_postfix(completed=sum(r["state"] == "completed" for r in records.values()))
    return sorted(
        records.values(),
        key=lambda record: (
            record.get("state") != "completed",
            record.get("selection_assd_macro_px") is None,
            record.get("selection_assd_macro_px", float("inf")),
        ),
    )


def make_progressive_baseline_configurations(
    template: ExperimentConfig,
) -> list[ExperimentConfig]:
    """Return the 12 baseline trials of the agreed progressive search.

    The baseline explores three learning rates, two U-Net widths, and two
    hardware-appropriate batch sizes.  The selected baseline and weighted
    finalist runs are intentionally separate notebook stages, so a loss
    comparison never changes architecture at the same time.
    """

    configurations: list[ExperimentConfig] = []
    for learning_rate in (1e-3, 3e-4, 1e-4):
        for base_channels in (16, 32):
            for batch_size in (2, 4):
                configurations.append(
                    replace(
                        template,
                        label=(
                            f"baseline_lr-{learning_rate:g}_width-{base_channels}_"
                            f"batch-{batch_size}"
                        ),
                        optimization=replace(
                            template.optimization,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                        ),
                        model=replace(template.model, base_channels=base_channels),
                    )
                )
    return configurations


def make_weighted_configurations(
    baseline_winner: ExperimentConfig,
) -> list[ExperimentConfig]:
    """Return eight weighted-CE trials with fixed winning architecture/batch size."""

    configurations: list[ExperimentConfig] = []
    for learning_rate in (3e-4, 1e-4):
        for alpha in (1.0, 3.0):
            for sigma in (2.0, 4.0):
                configurations.append(
                    replace(
                        baseline_winner,
                        label=(
                            f"weighted_lr-{learning_rate:g}_alpha-{alpha:g}_sigma-{sigma:g}"
                        ),
                        loss=replace(
                            baseline_winner.loss,
                            name="weighted_cross_entropy",
                            alpha=alpha,
                            sigma=sigma,
                        ),
                        optimization=replace(
                            baseline_winner.optimization,
                            learning_rate=learning_rate,
                        ),
                    )
                )
    return configurations


def make_finalist_repetitions(
    finalist: ExperimentConfig,
    *,
    seeds: tuple[int, int, int] = (42, 123, 2026),
) -> list[ExperimentConfig]:
    """Return the same final candidate for three comparable random seeds.

    The default seeds are reused unchanged for the CE and weighted-CE
    finalists, making their comparison independent of a different random
    initialization or loader order.
    """

    return [
        replace(
            finalist,
            label=f"{finalist.loss.name}_finalist_seed-{seed}",
            runtime=replace(finalist.runtime, seed=seed),
        )
        for seed in seeds
    ]
