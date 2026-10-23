"""Seed, deterministic-mode, and runtime-provenance helpers."""

from __future__ import annotations

import importlib.metadata
import os
import random
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA random sources.

    Deterministic kernels improve repeatability on a fixed software and
    hardware stack.  They cannot guarantee bit-for-bit equality across GPU,
    CUDA, driver, or PyTorch versions.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _initialise_worker(worker_id: int, *, seed: int) -> None:
    """Set worker-local random sources; kept top-level for Windows pickling."""

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def worker_init_fn(seed: int) -> Callable[[int], None]:
    """Return a DataLoader worker initializer derived from one experiment seed."""

    return partial(_initialise_worker, seed=seed)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve an explicit device request without silently falling back from CUDA."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested_device)


def git_commit(repository_root: str | Path) -> str | None:
    """Return the checked-out Git revision when the repository is available."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_provenance(repository_root: str | Path, device: torch.device) -> dict[str, Any]:
    """Collect execution metadata persisted with every experiment."""

    package_names = ["numpy", "scipy", "torch", "tifffile"]
    packages: dict[str, str] = {}
    for package_name in package_names:
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = "not-installed"
    return {
        "git_commit": git_commit(repository_root),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "packages": packages,
    }
