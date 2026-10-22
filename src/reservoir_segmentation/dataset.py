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
from collections.abc import Collection, Iterable
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
QUALITY_EXCLUSION_REQUIRED_COLUMNS = ("normalized_planetscope_path", "reason", "decision")
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


def _download_progress_message(downloaded: int, total: int | None, width: int = 30) -> str:
    """Format a compact terminal or notebook progress indicator."""

    downloaded_mib = downloaded / (1024 * 1024)
    if total is None or total <= 0:
        return f"\rDownloading RiverScope: {downloaded_mib:.1f} MiB received"
    fraction = min(downloaded / total, 1.0)
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    total_mib = total / (1024 * 1024)
    return (
        f"\rDownloading RiverScope: [{bar}] {fraction:.1%} "
        f"({downloaded_mib:.1f}/{total_mib:.1f} MiB)"
    )


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
            print(f"Using existing verified archive: {destination_path}")
            return destination_path
        raise ValueError(
            f"Existing archive has an unexpected MD5: {destination_path}. "
            "Remove it or choose another destination before retrying."
        )

    partial_path = destination_path.with_suffix(destination_path.suffix + ".part")
    try:
        print(f"Connecting to the official RiverScope archive: {url}", flush=True)
        with urllib.request.urlopen(url, timeout=180) as response, partial_path.open("wb") as out:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length and content_length.isdigit() else None
            update_interval = max(total // 100, chunk_size) if total else 8 * 1024 * 1024
            downloaded = 0
            next_update = 0
            while chunk := response.read(chunk_size):
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_update:
                    print(_download_progress_message(downloaded, total), end="", flush=True)
                    next_update = downloaded + update_interval
            print(_download_progress_message(downloaded, total), flush=True)
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


def _safe_planetscope_identifier(value: str) -> str:
    """Return an identifier that is safe to use in a generated filename or manifest."""

    identifier = value.strip()
    forbidden_characters = '<>:"/\\|?*'
    if (
        not identifier
        or identifier in {".", ".."}
        or any(character in identifier for character in forbidden_characters)
        or Path(identifier).name != identifier
    ):
        raise ValueError(f"Unsafe PlanetScope identifier: {value!r}")
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


def load_quality_exclusions(path: str | os.PathLike[str]) -> dict[str, str]:
    """Load the versioned tile-exclusion manifest as ``{image_path: reason}``.

    Only rows whose ``decision`` is ``exclude`` are returned. The official
    RiverScope CSVs and archive are not changed by this function.
    """

    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(QUALITY_EXCLUSION_REQUIRED_COLUMNS) - fieldnames
        if missing:
            raise ValueError(
                f"Quality-exclusion manifest is missing columns: {sorted(missing)}"
            )
        exclusions: dict[str, str] = {}
        for row in reader:
            if (row.get("decision") or "").strip().lower() != "exclude":
                continue
            image_path = _normalise_member(row.get("normalized_planetscope_path") or "")
            if not image_path or image_path in {".", ".."} or ".." in Path(image_path).parts:
                raise ValueError(f"Unsafe normalized PlanetScope path: {image_path!r}")
            reason = (row.get("reason") or "").strip()
            if not reason:
                raise ValueError(f"Quality-exclusion manifest has no reason for {image_path}")
            if image_path in exclusions:
                raise ValueError(f"Duplicate quality exclusion for {image_path}")
            exclusions[image_path] = reason
    return exclusions


def filter_riverscope_rows(
    rows: Iterable[dict[str, str]],
    excluded_image_paths: Collection[str],
) -> list[dict[str, str]]:
    """Return copies of rows not listed in a quality-exclusion manifest."""

    excluded_paths = {_normalise_member(path) for path in excluded_image_paths}
    return [
        dict(row)
        for row in rows
        if _normalise_member(row["normalized_planetscope_path"]) not in excluded_paths
    ]


def find_riverscope_all_zero_images(
    source: str | os.PathLike[str],
    split: str,
    *,
    progress_every: int = 50,
) -> list[dict[str, str | int]]:
    """Find valid-valued RiverScope images whose every band value is zero.

    The source may be an extracted dataset folder or the official ZIP. This
    function reports candidates; it never changes the source data or split
    CSVs. A candidate must be reviewed and recorded in the versioned quality
    manifest before derived workflows exclude it.
    """

    if progress_every < 0:
        raise ValueError("progress_every must not be negative")
    rows = read_riverscope_split(source, split)
    source_path = Path(source)
    candidates: list[dict[str, str | int]] = []
    archive = zipfile.ZipFile(source_path) if source_path.is_file() else None
    try:
        members = (
            [_normalise_member(name) for name in archive.namelist()] if archive is not None else []
        )
        for index, row in enumerate(rows, start=1):
            if progress_every and (index == 1 or index % progress_every == 0):
                print(f"Scanning {split}: {index}/{len(rows)}")
            image_reference = row["normalized_planetscope_path"]
            if archive is not None:
                image_member = resolve_riverscope_member(image_reference, members)
                with archive.open(image_member) as handle, tifffile.TiffFile(handle) as tif:
                    image = tif.asarray()
            else:
                local_image_path = _find_local_riverscope_file(source_path, image_reference)
                with tifffile.TiffFile(local_image_path) as tif:
                    image = tif.asarray()
            total_values = int(image.size)
            finite_values = int(np.isfinite(image).sum())
            nonzero_values = int(np.count_nonzero(image))
            if finite_values == total_values and nonzero_values == 0:
                candidate = dict(row)
                candidate.update(
                    total_values=total_values,
                    finite_values=finite_values,
                    nonzero_values=nonzero_values,
                )
                candidates.append(candidate)
    finally:
        if archive is not None:
            archive.close()
    return candidates


def materialize_riverscope_sample(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    split: str = "train",
    count: int = 6,
    seed: int = 2026,
    excluded_image_paths: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Copy a reproducible sample of paired PlanetScope tiles and labels.

    The returned rows retain every CSV field and add ``image_path`` and
    ``mask_path`` pointing to the materialized files. With a ZIP source only
    the selected pairs are extracted; the full archive remains untouched.
    """

    rows = filter_riverscope_rows(
        read_riverscope_split(source, split), excluded_image_paths
    )
    if count < 1:
        raise ValueError("count must be positive")
    if not rows:
        raise ValueError("No eligible rows remain after quality exclusions")
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
            identifier = _safe_planetscope_identifier(
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
