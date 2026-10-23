"""Manifest-driven RiverScope datasets and train-only normalization."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from reservoir_segmentation.dataset import resolve_riverscope_member, resolve_riverscope_path

from .reproducibility import worker_init_fn

MANIFEST_REQUIRED_COLUMNS = {
    "reach_id",
    "normalized_planetscope_path",
    "binary_mask_path",
    "valid_mask_path",
    "boundary_path",
    "distance_squared_path",
}


@dataclass(frozen=True)
class ChannelStatistics:
    """Per-channel statistics measured only over valid train pixels."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    valid_pixel_count: int

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.std) or not self.mean:
            raise ValueError("mean and std must be non-empty and have equal lengths")
        if any(value <= 0 for value in self.std):
            raise ValueError("every channel standard deviation must be positive")
        if self.valid_pixel_count < 1:
            raise ValueError("valid_pixel_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ChannelStatistics:
        """Restore statistics persisted by :meth:`to_dict`."""

        return cls(
            mean=tuple(float(value) for value in values["mean"]),
            std=tuple(float(value) for value in values["std"]),
            valid_pixel_count=int(values["valid_pixel_count"]),
        )


def read_geometry_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a geometry manifest without changing its published split assignment."""

    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Geometry manifest is empty: {manifest_path}")
    missing = MANIFEST_REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Geometry manifest is missing columns: {sorted(missing)}")
    return rows


def assert_split_isolation(manifests: Mapping[str, str | Path]) -> None:
    """Reject tile or geographic-group overlap between named published splits."""

    paths_by_split: dict[str, set[str]] = {}
    groups_by_split: dict[str, set[str]] = {}
    for split, path in manifests.items():
        rows = read_geometry_manifest(path)
        paths_by_split[split] = {row["normalized_planetscope_path"] for row in rows}
        groups_by_split[split] = {row["reach_id"] for row in rows}
    split_names = sorted(manifests)
    for index, first_split in enumerate(split_names):
        for second_split in split_names[index + 1 :]:
            duplicate_paths = paths_by_split[first_split] & paths_by_split[second_split]
            duplicate_groups = groups_by_split[first_split] & groups_by_split[second_split]
            if duplicate_paths or duplicate_groups:
                raise ValueError(
                    "Published split leakage between "
                    f"{first_split} and {second_split}: "
                    f"tiles={len(duplicate_paths)}, geographic_groups={len(duplicate_groups)}"
                )


def _to_channels_first(image: np.ndarray, expected_channels: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"PlanetScope image must be 3-D, got {array.shape}")
    if array.shape[0] == expected_channels:
        return array
    if array.shape[-1] == expected_channels:
        return np.moveaxis(array, -1, 0)
    raise ValueError(
        f"Expected {expected_channels} PlanetScope channels, got shape {array.shape}"
    )


def _absolute_map_path(geometry_root: Path, reference: str) -> Path:
    path = geometry_root / Path(reference)
    if not path.is_file():
        raise FileNotFoundError(f"Derived geometry map is missing: {path}")
    return path


def _stable_tile_id(row: Mapping[str, str]) -> str:
    reference = row["normalized_planetscope_path"]
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()[:12]


def _owned_tensor(array: np.ndarray, *, dtype: np.dtype[Any]) -> torch.Tensor:
    """Create a contiguous tensor with resizable, PyTorch-owned storage.

    A tensor returned directly by ``torch.from_numpy`` shares NumPy storage.
    On Windows, the default multi-worker DataLoader collate path uses shared
    storage and can reject those non-resizable tensors. Cloning after the
    conversion preserves the value and dtype while making every sample safe to
    batch in spawned workers.
    """

    contiguous = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return torch.from_numpy(contiguous).clone()


class RiverScopeSegmentationDataset(Dataset[dict[str, Any]]):
    """Read one fixed official split plus its aligned derived geometry maps.

    Labels and validity maps stay at their native spatial shape.  The U-Net
    pads only its image tensor internally and crops logits back to this shape,
    so neither targets nor metrics ever receive artificial padded pixels.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        source: str | Path,
        geometry_root: str | Path,
        *,
        statistics: ChannelStatistics | None = None,
        expected_channels: int = 4,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.source = Path(source)
        self.geometry_root = Path(geometry_root)
        self.rows = read_geometry_manifest(self.manifest_path)
        self.statistics = statistics
        self.expected_channels = expected_channels
        self._archive: zipfile.ZipFile | None = None
        self._archive_members: list[str] | None = None
        if statistics is not None and len(statistics.mean) != expected_channels:
            raise ValueError("statistics channel count does not match expected_channels")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = self._read_image(row).astype(np.float32, copy=False)
        target = np.load(_absolute_map_path(self.geometry_root, row["binary_mask_path"]))
        valid_mask = np.load(_absolute_map_path(self.geometry_root, row["valid_mask_path"]))
        boundary = np.load(_absolute_map_path(self.geometry_root, row["boundary_path"]))
        distance_squared = np.load(
            _absolute_map_path(self.geometry_root, row["distance_squared_path"])
        )
        expected_shape = tuple(image.shape[-2:])
        for name, array in {
            "target": target,
            "valid_mask": valid_mask,
            "boundary": boundary,
            "distance_squared": distance_squared,
        }.items():
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"{name} is not aligned with image "
                    f"{row['normalized_planetscope_path']}"
                )
        if self.statistics is not None:
            mean = np.asarray(self.statistics.mean, dtype=np.float32)[:, None, None]
            std = np.asarray(self.statistics.std, dtype=np.float32)[:, None, None]
            image = (image - mean) / std
        return {
            "image": _owned_tensor(image, dtype=np.float32),
            "target": _owned_tensor(target, dtype=np.int64),
            "valid_mask": _owned_tensor(valid_mask, dtype=np.bool_),
            "boundary": _owned_tensor(boundary, dtype=np.bool_),
            "distance_squared": _owned_tensor(distance_squared, dtype=np.float32),
            "tile_id": _stable_tile_id(row),
            "image_path": row["normalized_planetscope_path"],
            "geographic_group": row["reach_id"],
        }

    def _read_image(self, row: Mapping[str, str]) -> np.ndarray:
        """Read one image while keeping at most one ZIP handle per process."""

        reference = row["normalized_planetscope_path"]
        if self.source.is_dir():
            image_path = resolve_riverscope_path(self.source, reference)
            image = np.asarray(tifffile.imread(image_path))
        elif self.source.is_file() and self.source.suffix.lower() == ".zip":
            if self._archive is None:
                self._archive = zipfile.ZipFile(self.source)
                self._archive_members = self._archive.namelist()
            assert self._archive_members is not None
            member = resolve_riverscope_member(reference, self._archive_members)
            with self._archive.open(member) as handle:
                image = np.asarray(tifffile.imread(io.BytesIO(handle.read())))
        else:
            raise ValueError("source must be an extracted RiverScope directory or a ZIP archive")
        return _to_channels_first(image, self.expected_channels)

    def __getstate__(self) -> dict[str, Any]:
        """Avoid serializing an open archive when DataLoader spawns Windows workers."""

        state = dict(self.__dict__)
        state["_archive"] = None
        state["_archive_members"] = None
        return state

    def close(self) -> None:
        """Close the optional ZIP handle held by this process."""

        if self._archive is not None:
            self._archive.close()
            self._archive = None
            self._archive_members = None


def fit_channel_statistics(dataset: RiverScopeSegmentationDataset) -> ChannelStatistics:
    """Calculate per-channel train statistics over valid pixels only.

    The input dataset must represent the train manifest.  Passing a dataset
    with already-normalized images is rejected because that would hide source
    radiometry and could record circular statistics.
    """

    if dataset.statistics is not None:
        raise ValueError("fit_channel_statistics requires an unnormalized dataset")
    if dataset.manifest_path.stem != "train":
        raise ValueError("fit_channel_statistics accepts only the published train manifest")
    sums = np.zeros(dataset.expected_channels, dtype=np.float64)
    squared_sums = np.zeros(dataset.expected_channels, dtype=np.float64)
    valid_pixel_count = 0
    try:
        for row in dataset.rows:
            image = dataset._read_image(row).astype(np.float32, copy=False)
            valid = np.load(
                _absolute_map_path(dataset.geometry_root, row["valid_mask_path"])
            ).astype(bool, copy=False)
            if not valid.any():
                continue
            pixels = image[:, valid].astype(np.float64, copy=False)
            sums += pixels.sum(axis=1)
            squared_sums += np.square(pixels).sum(axis=1)
            valid_pixel_count += pixels.shape[1]
    finally:
        dataset.close()
    if valid_pixel_count < 2:
        raise ValueError("train manifest has fewer than two valid pixels")
    mean = sums / valid_pixel_count
    variance = squared_sums / valid_pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, np.finfo(np.float64).eps))
    return ChannelStatistics(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        valid_pixel_count=valid_pixel_count,
    )


def write_channel_statistics(statistics: ChannelStatistics, path: str | Path) -> None:
    """Persist train-only normalization statistics for reproducible inference."""

    Path(path).write_text(json.dumps(statistics.to_dict(), indent=2) + "\n", encoding="utf-8")


def collate_segmentation_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields without PyTorch's Windows shared-storage collation.

    The default collation path preallocates shared storage in each worker. On
    Windows this can fail for large RiverScope tiles even when the individual
    tensors own resizable storage. Explicit ``torch.stack`` keeps the batch
    construction predictable; strings remain a list so tile-level reporting
    preserves its one-value-per-sample convention.
    """

    if not samples:
        raise ValueError("cannot collate an empty segmentation batch")
    return {
        key: (
            torch.stack([sample[key] for sample in samples])
            if isinstance(value, torch.Tensor)
            else [sample[key] for sample in samples]
        )
        for key, value in samples[0].items()
    }


def build_dataloader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    """Create a seeded DataLoader with deterministic worker initialization."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn(seed),
        persistent_workers=num_workers > 0,
        collate_fn=collate_segmentation_batch,
    )


def select_overfit_indices(
    dataset: RiverScopeSegmentationDataset,
    *,
    count: int = 1,
) -> list[int]:
    """Choose valid train tiles nearest 50% water for the overfit smoke test.

    This is a diagnostic sampling helper, not a train/validation split or a
    model-selection operation. A balanced tile makes both classes observable
    in a short CPU smoke test and avoids treating an all-land tile as proof of
    successful segmentation.
    """

    if dataset.manifest_path.stem != "train":
        raise ValueError("select_overfit_indices accepts only the published train manifest")
    if count < 1:
        raise ValueError("count must be positive")
    candidates: list[tuple[float, int]] = []
    for index, row in enumerate(dataset.rows):
        target = np.load(_absolute_map_path(dataset.geometry_root, row["binary_mask_path"]))
        valid = np.load(_absolute_map_path(dataset.geometry_root, row["valid_mask_path"])).astype(
            bool, copy=False
        )
        if valid.any():
            water_fraction = float(np.asarray(target, dtype=bool)[valid].mean())
            candidates.append((abs(water_fraction - 0.5), index))
    if len(candidates) < count:
        raise ValueError("not enough valid train tiles for the requested smoke-test sample")
    return [index for _, index in sorted(candidates)[:count]]


def build_split_datasets(
    geometry_root: str | Path,
    source: str | Path,
    *,
    statistics: ChannelStatistics | None = None,
    include_test: bool = False,
) -> dict[str, RiverScopeSegmentationDataset]:
    """Build published train/validation datasets, optionally an explicit test set."""

    root = Path(geometry_root)
    split_names = ["train", "valid"]
    if include_test:
        split_names.append("test")
    manifests = {split: root / "manifests" / f"{split}.csv" for split in split_names}
    assert_split_isolation(manifests)
    return {
        split: RiverScopeSegmentationDataset(
            path,
            source,
            root,
            statistics=statistics,
        )
        for split, path in manifests.items()
    }
