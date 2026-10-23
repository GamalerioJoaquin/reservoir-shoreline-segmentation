"""Materialize geometry derived from RiverScope labels.

Usage example::

    python scripts/prepare_riverscope.py data/raw/RiverScope.zip \
        --output data/processed/riverscope/geometry-v2

The script never edits the archive, source GeoTIFFs, or published split CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from reservoir_segmentation.dataset import (
    filter_riverscope_rows,
    load_quality_exclusions,
    read_riverscope_split,
    resolve_riverscope_member,
    resolve_riverscope_path,
)
from reservoir_segmentation.geometry import (
    GEOMETRY_VERSION,
    RIVERSCOPE_IGNORED_LABELS,
    derive_riverscope_geometry,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUSIONS = REPOSITORY_ROOT / "configs" / "riverscope_quality_exclusions.csv"
DERIVED_COLUMNS = (
    "binary_mask_path",
    "valid_mask_path",
    "boundary_path",
    "distance_squared_path",
    "geometry_version",
    "ignored_label_3_pixels",
    "invalid_pixel_count",
)


def _tile_key(reference: str) -> str:
    normalised = reference.replace("\\", "/")
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]
    stem = Path(normalised).stem
    return f"{stem}-{digest}"


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.npy")
    try:
        np.save(temporary, array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _target_paths(output_root: Path, split: str, key: str) -> dict[str, Path]:
    return {
        "binary_mask_path": output_root / "binary_masks" / split / f"{key}.npy",
        "valid_mask_path": output_root / "valid_masks" / split / f"{key}.npy",
        "boundary_path": output_root / "boundaries" / split / f"{key}.npy",
        "distance_squared_path": output_root / "distance_squared" / split / f"{key}.npy",
    }


def _read_label(
    source: Path,
    reference: str,
    archive: zipfile.ZipFile | None,
    archive_members: list[str],
) -> np.ndarray:
    """Read one label from an extracted root or the official ZIP."""

    if archive is None:
        return tifffile.imread(resolve_riverscope_path(source, reference))
    member = resolve_riverscope_member(reference, archive_members)
    with archive.open(member) as handle, tifffile.TiffFile(handle) as tif:
        return tif.asarray()


def _validate_targets(records: list[dict[str, Any]], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        str(path)
        for record in records
        for path in record["targets"].values()
        if path.exists()
    ]
    if existing:
        example = "\n  ".join(existing[:3])
        raise FileExistsError(
            "Derived geometry already exists. Pass --overwrite to regenerate it.\n"
            f"  {example}"
        )


def _manifest_text(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def prepare_split(
    source: Path,
    output_root: Path,
    split: str,
    *,
    exclusions: dict[str, str],
    nodata_values: tuple[float, ...],
    overwrite: bool,
) -> dict[str, int]:
    """Generate all derived arrays and a manifest for one published split."""

    official_rows = read_riverscope_split(source, split)
    rows = filter_riverscope_rows(official_rows, exclusions)
    archive_context = zipfile.ZipFile(source) if source.is_file() else None
    archive_members = archive_context.namelist() if archive_context is not None else []
    records: list[dict[str, Any]] = []
    for row in rows:
        key = _tile_key(row["normalized_planetscope_path"])
        records.append(
            {
                "row": row,
                "targets": _target_paths(output_root, split, key),
            }
        )
    try:
        _validate_targets(records, overwrite=overwrite)

        manifest_rows: list[dict[str, str]] = []
        ignored_label_3_pixels = 0
        invalid_pixel_count = 0
        for index, record in enumerate(records, start=1):
            labels = _read_label(
                source,
                record["row"]["label_path"],
                archive_context,
                archive_members,
            )
            maps = derive_riverscope_geometry(labels, nodata_values=nodata_values, connectivity=8)
            targets = record["targets"]
            _atomic_save(targets["binary_mask_path"], maps.binary_mask)
            _atomic_save(targets["valid_mask_path"], maps.valid_mask)
            _atomic_save(targets["boundary_path"], maps.boundary)
            _atomic_save(targets["distance_squared_path"], maps.distance_squared)

            manifest_row = dict(record["row"])
            manifest_row.update(
                {
                    name: path.relative_to(output_root).as_posix()
                    for name, path in targets.items()
                }
            )
            manifest_row["geometry_version"] = GEOMETRY_VERSION
            manifest_row["ignored_label_3_pixels"] = str(np.count_nonzero(labels == 3))
            manifest_row["invalid_pixel_count"] = str(np.count_nonzero(~maps.valid_mask))
            manifest_rows.append(manifest_row)
            ignored_label_3_pixels += int(np.count_nonzero(labels == 3))
            invalid_pixel_count += int(np.count_nonzero(~maps.valid_mask))
            if index == 1 or index % 100 == 0 or index == len(records):
                progress = 100.0 * index / len(records)
                print(
                    f"{split}: processed {index}/{len(records)} tiles ({progress:.1f}%)",
                    flush=True,
                )

        fieldnames = [*official_rows[0].keys(), *DERIVED_COLUMNS]
        manifest_path = output_root / "manifests" / f"{split}.csv"
        _atomic_write_text(manifest_path, _manifest_text(manifest_rows, fieldnames))
    finally:
        if archive_context is not None:
            archive_context.close()
    return {
        "official": len(official_rows),
        "retained": len(rows),
        "excluded": len(official_rows) - len(rows),
        "ignored_label_3_pixels": ignored_label_3_pixels,
        "invalid_pixel_count": invalid_pixel_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Extracted RiverScope directory or the official RiverScope.zip archive.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Derived data destination.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=("train", "valid", "test"),
        help="Published splits to process; defaults to all three.",
    )
    parser.add_argument(
        "--quality-exclusions",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
        help="Versioned exclusion manifest to apply before deriving geometry.",
    )
    parser.add_argument(
        "--without-quality-exclusions",
        action="store_true",
        help="Do not apply the repository quality-exclusion manifest.",
    )
    parser.add_argument(
        "--label-nodata",
        action="append",
        type=float,
        default=[],
        metavar="VALUE",
        help="Additional invalid label value; may be specified more than once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived arrays and manifests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    args = build_parser().parse_args(argv)
    source = args.source.resolve()
    if not source.is_dir() and not (source.is_file() and source.suffix.lower() == ".zip"):
        raise SystemExit(
            "Source must be an extracted RiverScope directory or a .zip archive: "
            f"{source}"
        )
    output_root = args.output.resolve()
    exclusions = (
        {}
        if args.without_quality_exclusions
        else load_quality_exclusions(args.quality_exclusions)
    )
    nodata_values = (*RIVERSCOPE_IGNORED_LABELS, *args.label_nodata)
    summaries = {
        split: prepare_split(
            source,
            output_root,
            split,
            exclusions=exclusions,
            nodata_values=nodata_values,
            overwrite=args.overwrite,
        )
        for split in args.splits
    }
    metadata = {
        "geometry_version": GEOMETRY_VERSION,
        "boundary_convention": "thick 8-neighbour water/land transitions",
        "distance_representation": "squared Euclidean distance in pixels, float32",
        "label_policy": {"background": 0, "water": [1, 2], "ignored_no_data": [3]},
        "validity_policy": (
            "finite labels excluding the documented project policy value 3 and "
            "additional explicit --label-nodata values only; "
            "no cloud, shadow, or imagery-nodata inference"
        ),
        "label_nodata_values": list(nodata_values),
        "quality_exclusions": None
        if args.without_quality_exclusions
        else str(args.quality_exclusions.resolve()),
        "splits": summaries,
    }
    _atomic_write_text(
        output_root / "geometry_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
