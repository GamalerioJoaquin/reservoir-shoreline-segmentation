"""Synthetic checks for immutable published split safeguards."""

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from reservoir_segmentation.training.data import (
    RiverScopeSegmentationDataset,
    assert_split_isolation,
    build_dataloader,
)


def _write_manifest(path: Path, *, image_path: str, reach_id: str) -> None:
    fieldnames = [
        "reach_id",
        "normalized_planetscope_path",
        "binary_mask_path",
        "valid_mask_path",
        "boundary_path",
        "distance_squared_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "reach_id": reach_id,
                "normalized_planetscope_path": image_path,
                "binary_mask_path": "binary.npy",
                "valid_mask_path": "valid.npy",
                "boundary_path": "boundary.npy",
                "distance_squared_path": "distance.npy",
            }
        )


def test_split_isolation_rejects_shared_geographic_groups(tmp_path: Path):
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    _write_manifest(train, image_path="PlanetScope/input/train/a.tif", reach_id="reach-1")
    _write_manifest(valid, image_path="PlanetScope/input/valid/b.tif", reach_id="reach-1")

    with pytest.raises(ValueError, match="geographic_groups=1"):
        assert_split_isolation({"train": train, "valid": valid})


def test_dataset_batches_with_spawned_workers(tmp_path: Path):
    """NumPy-backed arrays must be safe for Windows DataLoader collation."""

    source_root = tmp_path / "source"
    image_reference = "PlanetScope/input/train/tile.tif"
    image_path = source_root / image_reference
    image_path.parent.mkdir(parents=True)
    tifffile.imwrite(
        image_path,
        np.full((3, 4, 4), 0.25, dtype=np.float32),
        photometric="minisblack",
    )

    geometry_root = tmp_path / "geometry"
    map_paths = {
        "binary_mask_path": "maps/binary.npy",
        "valid_mask_path": "maps/valid.npy",
        "boundary_path": "maps/boundary.npy",
        "distance_squared_path": "maps/distance_squared.npy",
    }
    for reference, array in {
        map_paths["binary_mask_path"]: np.array([[0, 1, 1, 0]] * 3, dtype=np.uint8),
        map_paths["valid_mask_path"]: np.ones((3, 4), dtype=bool),
        map_paths["boundary_path"]: np.ones((3, 4), dtype=bool),
        map_paths["distance_squared_path"]: np.zeros((3, 4), dtype=np.float32),
    }.items():
        path = geometry_root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    manifest = tmp_path / "train.csv"
    fieldnames = ["reach_id", "normalized_planetscope_path", *map_paths]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tile_number in range(2):
            writer.writerow(
                {
                    "reach_id": f"reach-{tile_number}",
                    "normalized_planetscope_path": image_reference,
                    **map_paths,
                }
            )

    dataset = RiverScopeSegmentationDataset(manifest, source_root, geometry_root)
    batch = next(
        iter(build_dataloader(dataset, batch_size=2, shuffle=False, seed=42, num_workers=2))
    )

    assert tuple(batch["image"].shape) == (2, 4, 3, 4)
    assert tuple(batch["target"].shape) == (2, 3, 4)
