"""Dataset discovery, pairing, metadata, and archive helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

RIVERSCOPE_RECORD_URL = "https://zenodo.org/records/15376394"
RIVERSCOPE_ARCHIVE_URL = "https://zenodo.org/records/15376394/files/RiverScope.zip"
RIVERSCOPE_ARCHIVE_MD5 = "910f3ffc046cbf60df30c27937555fa5"
RIVERSCOPE_REQUIRED_COLUMNS = (
    "reach_id",
    "mid_lat",
    "mid_lon",
    "planetscope_id",
    "normalized_planetscope_path",
    "label_path",
)
IMAGE_TOKENS = {"image", "images", "img", "rgb", "input", "inputs"}
MASK_TOKENS = {"label", "labels", "mask", "masks", "annotation", "annotations", "gt"}
TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetPair:
    """One image/mask pair discovered in a split."""

    split: str
    image_path: Path
    mask_path: Path


def _md5_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_riverscope_archive(
    destination: str | os.PathLike[str],
    *,
    url: str = RIVERSCOPE_ARCHIVE_URL,
    expected_md5: str = RIVERSCOPE_ARCHIVE_MD5,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download and verify the official RiverScope archive.

    The archive is roughly 8 GB. The function is intentionally explicit and
    never runs during notebook import. A partial download is kept separate
    from the final path and is removed when verification fails.
    """

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.is_file():
        if _md5_file(destination_path) == expected_md5:
            return destination_path
        raise ValueError(
            f"Existing archive has an unexpected MD5: {destination_path}. "
            "Remove it or choose another destination before retrying."
        )

    partial_path = destination_path.with_suffix(destination_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as response, partial_path.open("wb") as out:
            while chunk := response.read(chunk_size):
                out.write(chunk)
        actual_md5 = _md5_file(partial_path)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"RiverScope archive failed MD5 verification: expected {expected_md5}, "
                f"received {actual_md5}."
            )
        partial_path.replace(destination_path)
    finally:
        partial_path.unlink(missing_ok=True)
    return destination_path


def _normalise_member(path: str | os.PathLike[str]) -> str:
    return str(path).replace("\\", "/").lstrip("/").removeprefix("./")


def resolve_riverscope_member(reference: str, members: list[str]) -> str:
    """Resolve a CSV path against archive members, tolerating a root prefix."""

    normalised = _normalise_member(reference)
    candidates = [normalised]
    if normalised.startswith("RiverScope/"):
        candidates.append(normalised.removeprefix("RiverScope/"))
    matches = [
        member
        for candidate in candidates
        for member in members
        if member == candidate or member.endswith(f"/{candidate}")
    ]
    if not matches:
        raise FileNotFoundError(f"RiverScope member not found: {reference}")
    return min(matches, key=len)


def _find_local_riverscope_file(root: Path, reference: str) -> Path:
    normalised = Path(_normalise_member(reference))
    candidates = [root / normalised]
    if normalised.parts and normalised.parts[0].lower() == "riverscope":
        candidates.append(root / Path(*normalised.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = [path for path in root.rglob(normalised.name) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"RiverScope file not found: {reference}")
    return min(matches, key=lambda path: len(path.parts))


def _safe_sample_identifier(value: str) -> str:
    """Return an identifier that is safe to use as a generated filename."""

    identifier = value.strip()
    forbidden_characters = '<>:"/\\|?*'
    if (
        not identifier
        or identifier in {".", ".."}
        or any(character in identifier for character in forbidden_characters)
        or Path(identifier).name != identifier
    ):
        raise ValueError(f"Unsafe PlanetScope identifier for sample output: {value!r}")
    return identifier


def read_riverscope_split(
    source: str | os.PathLike[str],
    split: str = "train",
) -> list[dict[str, str]]:
    """Read an official RiverScope split CSV from a folder or ZIP archive."""

    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be one of: train, valid, test")
    source_path = Path(source)
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            members = [_normalise_member(name) for name in archive.namelist()]
            member = resolve_riverscope_member(f"{split}.csv", members)
            with archive.open(member) as handle:
                text_handle = io.TextIOWrapper(handle, encoding="utf-8-sig")
                rows = list(csv.DictReader(text_handle))
    else:
        csv_paths = sorted(source_path.rglob(f"{split}.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"Could not find {split}.csv below {source_path}")
        with csv_paths[0].open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    missing = set(RIVERSCOPE_REQUIRED_COLUMNS) - set(rows[0]) if rows else set()
    if missing:
        raise ValueError(f"RiverScope {split}.csv is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"RiverScope {split}.csv is empty")
    return rows


def materialize_riverscope_sample(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    split: str = "train",
    count: int = 6,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Copy a reproducible sample of paired PlanetScope tiles and labels.

    The returned rows retain every CSV field and add ``image_path`` and
    ``mask_path`` pointing to the materialized files. With a ZIP source only
    the selected pairs are extracted; the full archive remains untouched.
    """

    rows = read_riverscope_split(source, split)
    if count < 1:
        raise ValueError("count must be positive")
    selected = rows.copy()
    random.Random(seed).shuffle(selected)
    selected = selected[: min(count, len(selected))]
    source_path = Path(source)
    destination_path = Path(destination)
    results: list[dict[str, Any]] = []

    zip_context = zipfile.ZipFile(source_path) if source_path.is_file() else None
    try:
        members = (
            [_normalise_member(name) for name in zip_context.namelist()]
            if zip_context is not None
            else []
        )
        for row in selected:
            identifier = _safe_sample_identifier(
                row["planetscope_id"] or Path(row["normalized_planetscope_path"]).stem
            )
            image_target = destination_path / "images" / split / f"{identifier}.tif"
            mask_target = destination_path / "labels" / split / f"{identifier}.tif"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            image_ref = row["normalized_planetscope_path"]
            mask_ref = row["label_path"]
            if zip_context is not None:
                image_member = resolve_riverscope_member(image_ref, members)
                mask_member = resolve_riverscope_member(mask_ref, members)
                with (
                    zip_context.open(image_member) as source_handle,
                    image_target.open("wb") as target,
                ):
                    shutil.copyfileobj(source_handle, target)
                with (
                    zip_context.open(mask_member) as source_handle,
                    mask_target.open("wb") as target,
                ):
                    shutil.copyfileobj(source_handle, target)
            else:
                shutil.copyfile(_find_local_riverscope_file(source_path, image_ref), image_target)
                shutil.copyfile(_find_local_riverscope_file(source_path, mask_ref), mask_target)
            result = dict(row)
            result.update(image_path=image_target, mask_path=mask_target)
            results.append(result)
    finally:
        if zip_context is not None:
            zip_context.close()
    return results


def locate_dataset_root(
    explicit: str | os.PathLike[str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Return the first existing dataset root from explicit, env, or defaults."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_value = os.environ.get("RIVERSCOPE_DATA")
    if env_value:
        candidates.append(Path(env_value))
    base = Path(cwd) if cwd is not None else Path.cwd()
    candidates.extend(
        [
            base / "data" / "raw" / "riverscope",
            base / "data" / "raw",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _role(path: Path) -> str | None:
    tokens = {part.lower() for part in path.parts}
    if tokens & IMAGE_TOKENS:
        return "image"
    if tokens & MASK_TOKENS:
        return "mask"
    return None


def _key(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in (
        "_image",
        "-image",
        "_img",
        "-img",
        "_label",
        "-label",
        "_mask",
        "-mask",
        "_ann",
        "-ann",
        "_annotation",
        "-annotation",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def discover_pairs(root: str | os.PathLike[str]) -> list[DatasetPair]:
    """Pair TIFF images and masks by split and normalized filename.

    The function only pairs files when their directory names identify the role
    as image or mask. Ambiguous files are left out instead of guessed.
    """

    root_path = Path(root)
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        role = _role(path)
        if role is None:
            continue
        relative_parts = path.relative_to(root_path).parts
        split = next(
            (
                part.lower()
                for part in relative_parts[:-1]
                if part.lower() in {"train", "valid", "val", "test"}
            ),
            "unspecified",
        )
        grouped.setdefault((split, _key(path)), {})[role] = path

    pairs = []
    for (split, _), files in sorted(grouped.items()):
        if "image" in files and "mask" in files:
            pairs.append(
                DatasetPair(split=split, image_path=files["image"], mask_path=files["mask"])
            )
    return pairs


def read_tiff(path: str | os.PathLike[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a TIFF and return its array plus stable metadata for inventory tables."""

    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        array = tif.asarray()
        page = tif.pages[0]
        tags = page.tags
        metadata = {
            "path": str(path),
            "shape": tuple(int(value) for value in array.shape),
            "dtype": str(array.dtype),
            "ndim": int(array.ndim),
            "axes": getattr(page, "axes", None),
            "photometric": str(page.photometric) if page.photometric else None,
            "crs": _extract_crs(tif),
            "samples_per_pixel": int(tags["SamplesPerPixel"].value)
            if "SamplesPerPixel" in tags
            else None,
        }
    return np.asarray(array), metadata


def _extract_crs(tif: tifffile.TiffFile) -> str | None:
    """Best-effort CRS extraction without making geospatial assumptions."""

    xml = tif.pages[0].description or ""
    for marker in ("EPSG:", "EPSGCode="):
        start = xml.find(marker)
        if start >= 0:
            tail = xml[start:].split()[0].strip('";,<>')
            return tail
    return None


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Compute a streamed SHA-256 digest for duplicate detection."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_segmentation_archive(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> Path:
    """Safely extract a ZIP archive and return its destination directory."""

    archive_path = Path(archive)
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive_file:
        for member in archive_file.infolist():
            target = (destination_path / member.filename).resolve()
            if destination_path not in target.parents and target != destination_path:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive_file.extractall(destination_path)
    return destination_path
