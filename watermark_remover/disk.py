from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiskSpaceEstimate:
    scratch_bytes: int
    output_bytes: int


def estimate_disk_space(
    *,
    width: int,
    height: int,
    frame_count: int,
    input_size_bytes: int,
    save_debug: bool,
) -> DiskSpaceEstimate:
    """Return conservative scratch/output space estimates for one pipeline run."""
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions must be positive")
    if frame_count < 0:
        raise ValueError("Frame count must not be negative")
    if input_size_bytes < 0:
        raise ValueError("Input size must not be negative")

    frames = max(frame_count, 1)
    pixels = width * height * frames

    # Worst-case-ish uncompressed equivalents for the frame sets materialized by
    # the pipeline: source RGB, analytic RGB, residual mask, and inpainted RGB.
    bytes_per_pixel = 3 + 3 + 1 + 3
    if save_debug:
        # Alpha and confidence are stored as two additional grayscale images.
        bytes_per_pixel += 2

    # PNGs are usually smaller than their uncompressed form, but temporary files
    # created by downstream tools can add overhead. Keep a 25% safety margin.
    scratch_bytes = int(pixels * bytes_per_pixel * 1.25)

    # Reserve room for the encoded output and muxing overhead. Two input-file
    # equivalents is conservative for normal H.264 output while remaining useful
    # for low-bitrate source material.
    output_bytes = max(input_size_bytes * 2, 64 * 1024 * 1024)
    return DiskSpaceEstimate(scratch_bytes=scratch_bytes, output_bytes=output_bytes)


def ensure_pipeline_disk_space(
    scratch_path: Path,
    output_path: Path,
    estimate: DiskSpaceEstimate,
) -> None:
    """Validate scratch/output capacity without double-counting shared free space."""
    scratch_probe = _nearest_existing_path(scratch_path)
    output_probe = _nearest_existing_path(output_path)

    if _filesystem_device(scratch_probe) == _filesystem_device(output_probe):
        ensure_disk_space(
            scratch_probe,
            estimate.scratch_bytes + estimate.output_bytes,
            label="combined scratch/output",
        )
        return

    ensure_disk_space(scratch_probe, estimate.scratch_bytes, label="scratch")
    ensure_disk_space(output_probe, estimate.output_bytes, label="output")


def ensure_disk_space(path: Path, required_bytes: int, *, label: str) -> None:
    """Raise RuntimeError when the filesystem containing path lacks free space."""
    if required_bytes < 0:
        raise ValueError("Required disk space must not be negative")

    probe_path = _nearest_existing_path(path)
    free_bytes = shutil.disk_usage(probe_path).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient {label} disk space on {probe_path}: "
            f"requires {_format_bytes(required_bytes)}, "
            f"but only {_format_bytes(free_bytes)} is available."
        )


def _filesystem_device(path: Path) -> int:
    return os.stat(path).st_dev


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"Could not find an existing parent for {path}")
        candidate = parent
    return candidate


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")
