"""Train, validation, and inference loops for reproducible segmentation runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from reservoir_segmentation.models import UNet

from .config import ExperimentConfig, LossConfig, SelectionConfig
from .experiments import RunStore, load_checkpoint
from .losses import masked_cross_entropy, masked_weighted_cross_entropy
from .metrics import (
    aggregate_shoreline_by_group,
    merge_regional_metrics,
    regional_metrics,
    tile_metrics,
)
from .reproducibility import resolve_device, runtime_provenance, set_global_seed


@dataclass
class CheckpointSelector:
    """Keep a low-ASSD checkpoint while enforcing a validation-Dice safety band."""

    config: SelectionConfig
    best_epoch: int | None = None
    best_dice: float | None = None
    best_assd: float | None = None

    def consider(self, *, epoch: int, dice: float | None, assd: float | None) -> str | None:
        """Return an update reason when the supplied validation result becomes best."""

        if dice is None or assd is None or not math.isfinite(dice) or not math.isfinite(assd):
            return None
        if self.best_epoch is None:
            self._set_best(epoch, dice, assd)
            return "initial"
        assert self.best_dice is not None and self.best_assd is not None
        if dice < self.best_dice - self.config.dice_tolerance:
            return None
        if dice > self.best_dice + self.config.dice_tolerance:
            self._set_best(epoch, dice, assd)
            return "material_dice_improvement"
        if assd < self.best_assd - self.config.assd_min_delta:
            self._set_best(epoch, dice, assd)
            return "lower_assd_within_dice_band"
        return None

    def _set_best(self, epoch: int, dice: float, assd: float) -> None:
        self.best_epoch = epoch
        self.best_dice = dice
        self.best_assd = assd


def _loss_from_config(
    logits: Tensor,
    batch: Mapping[str, Any],
    loss_config: LossConfig,
) -> Tensor | None:
    if loss_config.name == "cross_entropy":
        return masked_cross_entropy(logits, batch["target"], batch["valid_mask"])
    assert loss_config.sigma is not None
    return masked_weighted_cross_entropy(
        logits,
        batch["target"],
        batch["valid_mask"],
        batch["distance_squared"],
        alpha=loss_config.alpha,
        sigma=loss_config.sigma,
    )


def _move_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _loader_contains_test(loader: DataLoader[Any]) -> bool:
    manifest_path = getattr(loader.dataset, "manifest_path", None)
    return manifest_path is not None and Path(manifest_path).stem == "test"


_CONFUSION_FIELDS = ("true_positive", "false_positive", "false_negative", "true_negative")


def _progress_postfix(
    total_loss: float,
    total_loss_pixels: int,
    regional_totals: Mapping[str, int],
    skipped_batches: int,
) -> dict[str, str]:
    """Format stable, epoch-to-date regional metrics for a tqdm postfix."""

    regional = merge_regional_metrics((regional_totals,))
    postfix = {
        "loss": f"{total_loss / total_loss_pixels:.4f}" if total_loss_pixels else "n/a",
        "dice": _format_metric(regional["dice"]),
        "iou": _format_metric(regional["iou"]),
        "acc": _format_metric(regional["masked_accuracy"]),
    }
    if skipped_batches:
        postfix["skipped"] = str(skipped_batches)
    return postfix


def _format_metric(value: float | int | None) -> str:
    """Render metrics consistently in progress output, including unavailable values."""

    return "n/a" if value is None else f"{float(value):.4f}"


def _format_epoch_summary(
    *,
    epoch: int,
    max_epochs: int,
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    selection_assd: float | None,
    learning_rate: float,
    checkpoint_update_reason: str | None,
    epochs_without_checkpoint_update: int | None,
    early_stopping_patience: int,
) -> str:
    """Return one readable, persistent summary line after a complete epoch."""

    def regional_text(result: Mapping[str, Any]) -> str:
        metrics = result["regional"]
        return (
            f"loss={_format_metric(result['loss'])} "
            f"dice={_format_metric(metrics['dice'])} "
            f"iou={_format_metric(metrics['iou'])} "
            f"acc={_format_metric(metrics['masked_accuracy'])}"
        )

    checkpoint_text = (
        f"checkpoint={checkpoint_update_reason}"
        if checkpoint_update_reason is not None
        else (
            "checkpoint_wait=n/a"
            if epochs_without_checkpoint_update is None
            else f"checkpoint_wait={epochs_without_checkpoint_update}/{early_stopping_patience}"
        )
    )
    return (
        f"epoch {epoch}/{max_epochs} | train {regional_text(training)} | "
        f"valid {regional_text(validation)} | val_assd={_format_metric(selection_assd)} | "
        f"next_lr={learning_rate:.2e} | {checkpoint_text}"
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    loss_config: LossConfig,
    optimizer: AdamW | None,
    use_mixed_precision: bool,
    progress_label: str,
    collect_tile_metrics: bool,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_loss_pixels = 0
    skipped_batches = 0
    skipped_tiles = 0
    regional_records: list[dict[str, float | int | None]] = []
    regional_totals = {field: 0 for field in _CONFUSION_FIELDS}
    tile_records: list[dict[str, Any]] = []
    amp_enabled = use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    context = torch.enable_grad if is_training else torch.no_grad
    batch_count = len(loader)
    progress_update_every = max(1, batch_count // 20)
    with context():
        progress = tqdm(loader, desc=progress_label, leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            device_batch = _move_tensors(batch, device)
            valid_pixels = int(device_batch["valid_mask"].sum().item())
            if valid_pixels == 0:
                skipped_batches += 1
                skipped_tiles += int(device_batch["image"].shape[0])
                if batch_index % progress_update_every == 0 or batch_index == batch_count:
                    progress.set_postfix(
                        _progress_postfix(
                            total_loss,
                            total_loss_pixels,
                            regional_totals,
                            skipped_batches,
                        )
                    )
                continue
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(device_batch["image"])
                loss = _loss_from_config(logits, device_batch, loss_config)
            if loss is None:
                skipped_batches += 1
                continue
            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach().item()) * valid_pixels
            total_loss_pixels += valid_pixels
            predictions = logits.detach().argmax(dim=1).cpu().numpy().astype(bool, copy=False)
            targets = device_batch["target"].detach().cpu().numpy().astype(bool, copy=False)
            validity = device_batch["valid_mask"].detach().cpu().numpy().astype(bool, copy=False)
            boundaries = device_batch["boundary"].detach().cpu().numpy().astype(bool, copy=False)
            for index in range(predictions.shape[0]):
                regional = regional_metrics(predictions[index], targets[index], validity[index])
                if regional["valid_pixels"] == 0:
                    skipped_tiles += 1
                    continue
                regional_records.append(regional)
                for field in _CONFUSION_FIELDS:
                    regional_totals[field] += int(regional[field])
                if collect_tile_metrics:
                    tile_records.append(
                        {
                            "tile_id": batch["tile_id"][index],
                            "image_path": batch["image_path"][index],
                            "geographic_group": batch["geographic_group"][index],
                            **tile_metrics(
                                predictions[index],
                                targets[index],
                                validity[index],
                                reference_boundary=boundaries[index],
                            ),
                        }
                    )
            if batch_index % progress_update_every == 0 or batch_index == batch_count:
                progress.set_postfix(
                    _progress_postfix(
                        total_loss,
                        total_loss_pixels,
                        regional_totals,
                        skipped_batches,
                    )
                )
    global_regional = merge_regional_metrics(regional_records)
    shoreline = aggregate_shoreline_by_group(tile_records) if collect_tile_metrics else None
    return {
        "loss": total_loss / total_loss_pixels if total_loss_pixels else None,
        "regional": global_regional,
        "shoreline": shoreline,
        "skipped_invalid_batches": skipped_batches,
        "skipped_invalid_tiles": skipped_tiles,
        "tiles": tile_records,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_config: LossConfig,
    *,
    device: str | torch.device = "auto",
    progress_label: str = "evaluation",
) -> dict[str, Any]:
    """Run inference and metrics without using results for optimization or tuning."""

    target_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    model = model.to(target_device)
    return _run_epoch(
        model,
        loader,
        device=target_device,
        loss_config=loss_config,
        optimizer=None,
        use_mixed_precision=False,
        progress_label=progress_label,
        collect_tile_metrics=True,
    )


def run_training(
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    channel_statistics: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    """Train on train, select exclusively on validation, and restore best weights.

    Test manifests are rejected here so model selection and grid search cannot
    accidentally consume them.  Test evaluation is a separate explicit call
    to :func:`evaluate` after a final configuration is frozen.
    """

    if _loader_contains_test(train_loader) or _loader_contains_test(validation_loader):
        raise ValueError("run_training accepts only train and validation loaders, never test")
    set_global_seed(config.runtime.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    model = UNet(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        encoder_depth=config.model.encoder_depth,
        padding_mode=config.model.padding_mode,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.optimization.scheduler_factor,
        patience=config.optimization.scheduler_patience,
        threshold=config.optimization.scheduler_min_delta,
        threshold_mode="abs",
        min_lr=config.optimization.min_learning_rate,
    )
    provenance = runtime_provenance(repository_root, device)
    provenance["seed"] = config.runtime.seed
    provenance["deterministic"] = config.runtime.deterministic
    store = RunStore(
        output_dir,
        config,
        channel_statistics=channel_statistics,
        provenance=provenance,
    )
    selector = CheckpointSelector(config.selection)
    history: list[dict[str, Any]] = []
    try:
        for epoch in range(1, config.optimization.max_epochs + 1):
            training = _run_epoch(
                model,
                train_loader,
                device=device,
                loss_config=config.loss,
                optimizer=optimizer,
                use_mixed_precision=config.runtime.mixed_precision,
                progress_label=f"train {epoch}/{config.optimization.max_epochs}",
                collect_tile_metrics=False,
            )
            validation = _run_epoch(
                model,
                validation_loader,
                device=device,
                loss_config=config.loss,
                optimizer=None,
                use_mixed_precision=config.runtime.mixed_precision,
                progress_label=f"validation {epoch}/{config.optimization.max_epochs}",
                collect_tile_metrics=True,
            )
            selection_assd = validation["shoreline"]["selection_assd_macro_px"]
            validation_dice = validation["regional"]["dice"]
            scheduler.step(selection_assd if selection_assd is not None else float("inf"))
            update_reason = selector.consider(
                epoch=epoch,
                dice=validation_dice,
                assd=selection_assd,
            )
            epoch_metrics = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training": {key: value for key, value in training.items() if key != "tiles"},
                "validation": {key: value for key, value in validation.items() if key != "tiles"},
                "checkpoint_update_reason": update_reason,
            }
            store.append_epoch(epoch_metrics)
            history.append(epoch_metrics)
            if update_reason is not None:
                store.save_best_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "experiment_config": config.to_dict(),
                        "channel_statistics": dict(channel_statistics),
                        "validation": epoch_metrics["validation"],
                    }
                )
            epochs_without_checkpoint_update = (
                None if selector.best_epoch is None else epoch - selector.best_epoch
            )
            tqdm.write(
                _format_epoch_summary(
                    epoch=epoch,
                    max_epochs=config.optimization.max_epochs,
                    training=training,
                    validation=validation,
                    selection_assd=selection_assd,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    checkpoint_update_reason=update_reason,
                    epochs_without_checkpoint_update=epochs_without_checkpoint_update,
                    early_stopping_patience=config.selection.early_stopping_patience,
                )
            )
            if selector.best_epoch is not None and (
                epoch - selector.best_epoch >= config.selection.early_stopping_patience
            ):
                break
        if selector.best_epoch is None:
            raise RuntimeError("No checkpoint was eligible; validation had no valid metrics")
        checkpoint = load_checkpoint(store.paths.best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        restored_validation = evaluate(
            model,
            validation_loader,
            config.loss,
            device=device,
            progress_label="best validation",
        )
        summary = {
            "state": "completed",
            "best_epoch": selector.best_epoch,
            "best_validation_dice": selector.best_dice,
            "best_validation_selection_assd_macro_px": selector.best_assd,
            "epochs_completed": len(history),
            "early_stopping_patience": config.selection.early_stopping_patience,
            "restored_validation": {
                key: value for key, value in restored_validation.items() if key != "tiles"
            },
        }
        save_training_curves(history, store.paths.curves)
        store.complete(summary)
        return {
            "model": model,
            "summary": summary,
            "history": history,
            "paths": store.paths,
            "best_checkpoint": store.paths.best_checkpoint,
        }
    except BaseException as error:
        store.fail(error)
        raise


def overfit_smoke_test(
    train_loader: DataLoader[Any],
    config: ExperimentConfig,
    *,
    steps: int = 100,
) -> dict[str, float | int]:
    """Overfit one deterministic train batch before a full search is started.

    This intentionally uses no validation or test data.  It is a diagnostic
    for data/target alignment, gradient flow, masking, and model capacity;
    it is not an experiment result and is never persisted as a candidate.
    """

    if steps < 2:
        raise ValueError("steps must be at least two")
    if _loader_contains_test(train_loader):
        raise ValueError("overfit_smoke_test accepts only a train loader")
    set_global_seed(config.runtime.seed, deterministic=config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    model = UNet(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        encoder_depth=config.model.encoder_depth,
        padding_mode=config.model.padding_mode,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=0.0,
    )
    batch = _move_tensors(next(iter(train_loader)), device)
    if int(batch["valid_mask"].sum().item()) == 0:
        raise RuntimeError("smoke-test batch has no valid pixels")
    losses: list[float] = []
    progress = tqdm(range(steps), desc="overfit smoke test", leave=False)
    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"])
        loss = _loss_from_config(logits, batch, config.loss)
        if loss is None:
            raise RuntimeError("smoke-test batch became invalid")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")
    with torch.no_grad():
        prediction = model(batch["image"]).argmax(dim=1).cpu().numpy().astype(bool, copy=False)
    target = batch["target"].cpu().numpy().astype(bool, copy=False)
    valid = batch["valid_mask"].cpu().numpy().astype(bool, copy=False)
    dice = regional_metrics(prediction[0], target[0], valid[0])["dice"]
    return {
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "final_first_tile_dice": float(dice) if dice is not None else float("nan"),
    }


def model_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[UNet, ExperimentConfig, dict[str, Any]]:
    """Recreate the frozen model and configuration selected before test evaluation."""

    target_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    checkpoint = load_checkpoint(path, map_location=target_device)
    config = ExperimentConfig.from_dict(checkpoint["experiment_config"])
    model = UNet(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        encoder_depth=config.model.encoder_depth,
        padding_mode=config.model.padding_mode,
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def save_training_curves(history: list[Mapping[str, Any]], path: str | Path) -> None:
    """Save train/validation loss and regional learning curves for one run."""

    if not history:
        raise ValueError("cannot plot an empty training history")
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, [row["training"]["loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation"]["loss"] for row in history], label="validation")
    axes[0].set(title="Masked loss", xlabel="Epoch")
    axes[0].legend()
    for metric in ("dice", "iou", "masked_accuracy"):
        axes[1].plot(
            epochs,
            [row["training"]["regional"][metric] for row in history],
            label=f"train {metric}",
        )
        axes[1].plot(
            epochs,
            [row["validation"]["regional"][metric] for row in history],
            linestyle="--",
            label=f"validation {metric}",
        )
    axes[1].set(title="Regional metrics", xlabel="Epoch")
    axes[1].legend(fontsize=7)
    axes[2].plot(
        epochs,
        [row["validation"]["shoreline"]["selection_assd_macro_px"] for row in history],
    )
    axes[2].set(title="Validation selection ASSD", xlabel="Epoch", ylabel="Pixels")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
