"""Synthetic checks for immutable published split safeguards."""

import csv
from pathlib import Path

import pytest

from reservoir_segmentation.training.data import assert_split_isolation


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
