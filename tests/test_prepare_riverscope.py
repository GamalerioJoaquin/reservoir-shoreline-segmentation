import csv
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile


def test_prepare_script_writes_geometry_maps_and_manifest(tmp_path: Path):
    source = tmp_path / "riverscope"
    label = source / "PlanetScope" / "label" / "train" / "sample.tif"
    image = source / "PlanetScope" / "input" / "train" / "sample.tif"
    label.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    tifffile.imwrite(label, np.array([[0, 1], [2, 3]], dtype=np.uint8))
    tifffile.imwrite(image, np.zeros((2, 2, 4), dtype=np.uint8))

    fields = [
        "reach_id",
        "mid_lat",
        "mid_lon",
        "planetscope_id",
        "normalized_planetscope_path",
        "label_path",
    ]
    with (source / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "reach_id": "1",
                "mid_lat": "-34.6",
                "mid_lon": "-58.4",
                "planetscope_id": "sample",
                "normalized_planetscope_path": "PlanetScope/input/train/sample.tif",
                "label_path": "PlanetScope/label/train/sample.tif",
            }
        )

    output = tmp_path / "processed"
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_riverscope.py",
            str(source),
            "--output",
            str(output),
            "--splits",
            "train",
            "--without-quality-exclusions",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with (output / "manifests" / "train.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    distance_squared = np.load(output / row["distance_squared_path"])

    assert np.load(output / row["binary_mask_path"]).dtype == bool
    assert not np.load(output / row["valid_mask_path"])[1, 1]
    assert np.load(output / row["boundary_path"]).any()
    assert distance_squared.dtype == np.float32
    assert row["ignored_label_3_pixels"] == "1"
    assert row["invalid_pixel_count"] == "1"
    assert (output / "geometry_metadata.json").is_file()

    archive = tmp_path / "RiverScope.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(source))
    zip_output = tmp_path / "processed-from-zip"
    zip_result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_riverscope.py",
            str(archive),
            "--output",
            str(zip_output),
            "--splits",
            "train",
            "--without-quality-exclusions",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert zip_result.returncode == 0, zip_result.stderr
    assert (zip_output / "manifests" / "train.csv").is_file()
