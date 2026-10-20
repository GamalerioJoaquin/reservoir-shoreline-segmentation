"""Reusable utilities for reservoir shoreline segmentation."""

from .boundaries import (
    distance_squared_to_boundary,
    find_boundary_pixels,
    weights_from_distance_squared,
)
from .dataset import (
    RIVERSCOPE_ARCHIVE_MD5,
    RIVERSCOPE_ARCHIVE_URL,
    RIVERSCOPE_RECORD_URL,
    DatasetPair,
    discover_pairs,
    download_riverscope_archive,
    locate_dataset_root,
    materialize_riverscope_sample,
    read_riverscope_split,
    read_tiff,
    resolve_riverscope_member,
    sha256_file,
)

__all__ = [
    "RIVERSCOPE_ARCHIVE_MD5",
    "RIVERSCOPE_ARCHIVE_URL",
    "RIVERSCOPE_RECORD_URL",
    "DatasetPair",
    "discover_pairs",
    "distance_squared_to_boundary",
    "download_riverscope_archive",
    "find_boundary_pixels",
    "locate_dataset_root",
    "materialize_riverscope_sample",
    "read_riverscope_split",
    "read_tiff",
    "resolve_riverscope_member",
    "sha256_file",
    "weights_from_distance_squared",
]
