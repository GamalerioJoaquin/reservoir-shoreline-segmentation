"""Persistent, inspectable experiment and grid-search artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def canonical_config_hash(config: ExperimentConfig) -> str:
    """Return a stable identity for a complete experiment configuration."""

    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a readable JSON document."""

    _atomic_write_text(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
    )


@dataclass(frozen=True)
class RunPaths:
    """Locations generated for one reproducible training execution."""

    root: Path
    config: Path
    provenance: Path
    history: Path
    curves: Path
    best_checkpoint: Path
    summary: Path
    status: Path


class RunStore:
    """Write metrics after every epoch and preserve the best checkpoint."""

    def __init__(
        self,
        root: str | Path,
        config: ExperimentConfig,
        *,
        channel_statistics: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> None:
        run_root = Path(root)
        run_root.mkdir(parents=True, exist_ok=True)
        self.paths = RunPaths(
            root=run_root,
            config=run_root / "run_config.json",
            provenance=run_root / "provenance.json",
            history=run_root / "metrics.jsonl",
            curves=run_root / "training_curves.png",
            best_checkpoint=run_root / "best_checkpoint.pt",
            summary=run_root / "summary.json",
            status=run_root / "status.json",
        )
        write_json(
            self.paths.config,
            {
                "config_hash": canonical_config_hash(config),
                "experiment_config": config.to_dict(),
                "channel_statistics": dict(channel_statistics),
            },
        )
        write_json(self.paths.provenance, dict(provenance))
        write_json(self.paths.status, {"state": "running"})

    def append_epoch(self, metrics: Mapping[str, Any]) -> None:
        """Append one fully evaluated epoch and force it to durable storage."""

        self.paths.history.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save_best_checkpoint(self, payload: Mapping[str, Any]) -> None:
        """Atomically replace the checkpoint selected from validation metrics."""

        temporary_path = self.paths.best_checkpoint.with_suffix(".pt.tmp")
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, self.paths.best_checkpoint)

    def complete(self, summary: Mapping[str, Any]) -> None:
        """Persist final run summary and state."""

        write_json(self.paths.summary, dict(summary))
        write_json(self.paths.status, {"state": "completed"})

    def fail(self, error: BaseException) -> None:
        """Persist failure state without discarding previous epoch artifacts."""

        write_json(
            self.paths.status,
            {"state": "failed", "error_type": type(error).__name__, "message": str(error)},
        )


class GridSearchStore:
    """Persist grid-search trial state so completed configurations are skipped."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.trials_root = self.root / "trials"
        self.results_path = self.root / "results.csv"
        self.winner_path = self.root / "winner.json"
        self.trials_root.mkdir(parents=True, exist_ok=True)

    def trial_root(self, config: ExperimentConfig) -> Path:
        """Return the deterministic directory for a single configuration."""

        return self.trials_root / canonical_config_hash(config)

    def is_completed(self, config: ExperimentConfig) -> bool:
        """Return whether the matching trial completed successfully."""

        status_path = self.trial_root(config) / "status.json"
        if not status_path.is_file():
            return False
        return json.loads(status_path.read_text(encoding="utf-8")).get("state") == "completed"

    def write_results(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Replace the sorted, tabular summary after every finished trial."""

        rows = [dict(record) for record in records]
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row})
        temporary_path = self.results_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, self.results_path)

    def write_winner(self, record: Mapping[str, Any]) -> None:
        """Persist the selected configuration explicitly, never inferred from test."""

        write_json(self.winner_path, dict(record))


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a checkpoint saved by :class:`RunStore`."""

    return torch.load(Path(path), map_location=map_location, weights_only=False)
