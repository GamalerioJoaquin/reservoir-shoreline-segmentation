import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from reservoir_segmentation.boundaries import (
    distance_squared_to_boundary,
    find_boundary_pixels,
    weights_from_distance_squared,
)
from reservoir_segmentation.dataset import (
    _download_progress_message,
    discover_pairs,
    filter_riverscope_rows,
    find_riverscope_all_zero_images,
    load_quality_exclusions,
    locate_dataset_root,
    materialize_riverscope_rows,
    materialize_riverscope_sample,
    read_riverscope_split,
)
from reservoir_segmentation.visualization import crop_to_overlap, overlay_mask


def test_discover_pairs_by_split_and_role(tmp_path: Path):
    for split in ("train", "test"):
        image_dir = tmp_path / split / "images"
        mask_dir = tmp_path / split / "labels"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        tifffile.imwrite(image_dir / "sample.tif", np.zeros((4, 5, 3), dtype=np.uint8))
        tifffile.imwrite(mask_dir / "sample-ann.tif", np.ones((4, 5), dtype=np.uint8))

    pairs = discover_pairs(tmp_path)

    assert [(pair.split, pair.image_path.stem) for pair in pairs] == [
        ("test", "sample"),
        ("train", "sample"),
    ]


def test_locate_dataset_prefers_explicit_path(tmp_path: Path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)

    assert locate_dataset_root(explicit, cwd=tmp_path) == explicit.resolve()


def test_distance_and_weights_emphasize_boundary():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    boundary = find_boundary_pixels(mask)
    distance_squared = distance_squared_to_boundary(boundary)
    weights = weights_from_distance_squared(distance_squared, sigma=2.0)

    assert boundary.any()
    assert distance_squared[2, 2] == 0
    assert weights[2, 2] > weights[0, 0]


def test_download_progress_message_reports_known_and_unknown_sizes():
    assert "50.0%" in _download_progress_message(5 * 1024 * 1024, 10 * 1024 * 1024)
    assert "5.0 MiB received" in _download_progress_message(5 * 1024 * 1024, None)


def test_visualization_crops_and_overlays():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 0] = 1

    cropped_image, cropped_mask = crop_to_overlap(image, mask)
    output = overlay_mask(image, mask)

    assert cropped_image.shape == (2, 3, 3)
    assert cropped_mask.shape == (2, 3)
    assert output.shape == (2, 3, 3)
    assert output[0, 0, 0] > 0


def test_unet_returns_two_probability_maps_at_original_tile_size():
    torch = pytest.importorskip("torch")
    from reservoir_segmentation.models import UNet, class_labels, expected_output_shape

    model = UNet(in_channels=4, base_channels=2)
    images = torch.zeros((1, 4, 500, 500))

    logits = model.forward_logits(images)
    probabilities = model(images)

    assert tuple(logits.shape) == (1, 2, 500, 500)
    assert tuple(probabilities.shape) == tuple(expected_output_shape(batch_size=1))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones((1, 500, 500)))
    assert tuple(class_labels(probabilities).shape) == (1, 500, 500)


def test_unet_rejects_non_riverscope_tile_shapes():
    torch = pytest.importorskip("torch")
    from reservoir_segmentation.models import UNet

    model = UNet(in_channels=4, base_channels=2)

    with pytest.raises(ValueError, match="Expected 500x500 inputs"):
        model(torch.zeros((1, 4, 512, 512)))


def test_riverscope_split_and_sample_from_zip(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(image_path, np.zeros((4, 5, 4), dtype=np.uint16))
    tifffile.imwrite(mask_path, np.ones((4, 5), dtype=np.uint8))
    csv_path = tmp_path / "train.csv"
    fields = [
        "reach_id",
        "mid_lat",
        "mid_lon",
        "planetscope_id",
        "normalized_planetscope_path",
        "label_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "reach_id": "7",
                "mid_lat": "-34.6",
                "mid_lon": "-58.4",
                "planetscope_id": "tile-1",
                "normalized_planetscope_path": "RiverScope/PlanetScope/input/train/image.tif",
                "label_path": "RiverScope/PlanetScope/label/train/mask.tif",
            }
        )
    archive_path = tmp_path / "RiverScope.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, "RiverScope/train.csv")
        archive.write(image_path, "RiverScope/PlanetScope/input/train/image.tif")
        archive.write(mask_path, "RiverScope/PlanetScope/label/train/mask.tif")

    rows = read_riverscope_split(archive_path, "train")
    sample = materialize_riverscope_sample(archive_path, tmp_path / "sample", count=1)
    zero_images = find_riverscope_all_zero_images(archive_path, "train", progress_every=0)

    assert rows[0]["reach_id"] == "7"
    assert sample[0]["planetscope_id"] == "tile-1"
    assert sample[0]["image_path"].is_file()
    assert sample[0]["mask_path"].is_file()
    assert [row["planetscope_id"] for row in zero_images] == ["tile-1"]


def test_materialize_riverscope_sample_rejects_unsafe_reference(tmp_path: Path):
    archive_path = tmp_path / "unsafe.zip"
    csv_text = (
        "reach_id,mid_lat,mid_lon,planetscope_id,normalized_planetscope_path,label_path\n"
        "7,-34.6,-58.4,tile-1,../../../outside.tif,"
        "PlanetScope/label/train/mask.tif\n"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("RiverScope/train.csv", csv_text)

    with pytest.raises(ValueError, match="Unsafe RiverScope path"):
        materialize_riverscope_sample(archive_path, tmp_path / "sample", count=1)


def test_materialize_riverscope_rows_uses_unique_tile_paths(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(image_path, np.zeros((4, 5, 4), dtype=np.uint16))
    tifffile.imwrite(mask_path, np.ones((4, 5), dtype=np.uint8))
    archive_path = tmp_path / "RiverScope.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for suffix in ("crop-a", "crop-b"):
            archive.write(image_path, f"RiverScope/PlanetScope/input/train/{suffix}.tif")
            archive.write(mask_path, f"RiverScope/PlanetScope/label/train/{suffix}.tif")

    rows = [
        {
            "planetscope_id": "shared-scene",
            "normalized_planetscope_path": f"PlanetScope/input/train/{suffix}.tif",
            "label_path": f"PlanetScope/label/train/{suffix}.tif",
        }
        for suffix in ("crop-a", "crop-b")
    ]

    materialized = materialize_riverscope_rows(archive_path, tmp_path / "sample", rows)

    image_names = [row["image_path"].name for row in materialized]

    assert image_names[0].startswith("crop-a-")
    assert image_names[1].startswith("crop-b-")
    assert len(set(image_names)) == 2
    assert all(row["image_path"].is_file() for row in materialized)


def test_quality_exclusion_manifest_filters_rows(tmp_path: Path):
    manifest = tmp_path / "quality_exclusions.csv"
    manifest.write_text(
        "normalized_planetscope_path,reason,decision\n"
        "PlanetScope/input/train/bad-tile.tif,all_zero_imagery,exclude\n"
        "PlanetScope/input/train/review-tile.tif,manual_review,retain\n",
        encoding="utf-8",
    )
    rows = [
        {"normalized_planetscope_path": "PlanetScope/input/train/bad-tile.tif"},
        {"normalized_planetscope_path": "PlanetScope/input/train/good-tile.tif"},
    ]

    exclusions = load_quality_exclusions(manifest)

    assert exclusions == {"PlanetScope/input/train/bad-tile.tif": "all_zero_imagery"}
    assert filter_riverscope_rows(rows, exclusions) == [
        {"normalized_planetscope_path": "PlanetScope/input/train/good-tile.tif"}
    ]
