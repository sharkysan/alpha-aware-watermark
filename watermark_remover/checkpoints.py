from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunks import FrameChunk
from .models import PipelineConfig, QualityMetrics


class CheckpointStore:
    """Persistent analytic/residual chunk checkpoints for interrupted runs."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.root = _checkpoint_root(config)
        self.analytic_dir = self.root / "analytic"
        self.residual_dir = self.root / "residual_masks"
        self.debug_dir = self.root / "debug"
        self.chunk_dir = self.root / "chunks"
        self.manifest_path = self.root / "manifest.json"
        self.fingerprint = build_checkpoint_fingerprint(config)

    def prepare(self) -> None:
        """Create a compatible store, discarding stale incompatible state."""
        if self.root.exists() and not self._manifest_matches():
            shutil.rmtree(self.root)

        self.analytic_dir.mkdir(parents=True, exist_ok=True)
        self.residual_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def load_chunk(self, chunk: FrameChunk) -> list[QualityMetrics] | None:
        """Return saved metrics only when all required chunk outputs exist."""
        record_path = self._chunk_record_path(chunk)
        if not record_path.is_file():
            return None

        for index in range(chunk.process_start, chunk.process_end):
            if not _nonempty_file(self.analytic_dir / f"{index:05d}.png"):
                return None
            if not _nonempty_file(self.residual_dir / f"{index:05d}.png"):
                return None

        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            raw_metrics = payload["metrics"]
            if not isinstance(raw_metrics, list):
                return None
            metrics = [QualityMetrics(**item) for item in raw_metrics]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

        expected = list(range(chunk.process_start, chunk.process_end))
        if [item.frame_index for item in metrics] != expected:
            return None
        return metrics

    def save_chunk(self, chunk: FrameChunk, metrics: list[QualityMetrics]) -> None:
        """Atomically mark a chunk complete after its output files are written."""
        payload = {
            "process_start": chunk.process_start,
            "process_end": chunk.process_end,
            "metrics": [asdict(item) for item in metrics],
        }
        record_path = self._chunk_record_path(chunk)
        temp_path = record_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temp_path.replace(record_path)

    def cleanup(self) -> None:
        """Remove checkpoint data after a fully successful pipeline run."""
        if self.root.exists():
            shutil.rmtree(self.root)

    def _manifest_matches(self) -> bool:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("fingerprint") == self.fingerprint

    def _write_manifest(self) -> None:
        payload = {"version": 1, "fingerprint": self.fingerprint}
        temp_path = self.manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.manifest_path)

    def _chunk_record_path(self, chunk: FrameChunk) -> Path:
        return self.chunk_dir / f"{chunk.process_start:06d}-{chunk.process_end:06d}.json"


def build_checkpoint_fingerprint(config: PipelineConfig) -> str:
    """Hash source identity and settings that affect reusable chunk outputs."""
    input_stat = config.input_path.stat()
    payload: dict[str, Any] = {
        "input": {
            "path": str(config.input_path.expanduser().resolve()),
            "size": input_stat.st_size,
            "mtime_ns": input_stat.st_mtime_ns,
        },
        "mask_dir": _mask_directory_identity(config.mask_dir),
        "region": asdict(config.region) if config.region is not None else None,
        "temporal_radius": config.temporal_radius,
        "chunk_size": config.chunk_size,
        "scene_threshold": config.scene_threshold,
        "min_scene_length": config.min_scene_length,
        "motion_compensation": config.motion_compensation,
        "alpha_inpaint_threshold": config.alpha_inpaint_threshold,
        "analytic_confidence_min": config.analytic_confidence_min,
        "residual_dilate": config.residual_dilate,
        "save_debug": config.save_debug,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_root(config: PipelineConfig) -> Path:
    if config.checkpoint_dir is not None:
        return config.checkpoint_dir.expanduser()
    return config.output_path.parent / ".alpha_wm_checkpoints" / config.input_path.stem


def _mask_directory_identity(mask_dir: Path | None) -> list[dict[str, int | str]] | None:
    if mask_dir is None:
        return None
    root = mask_dir.expanduser().resolve()
    if not root.exists():
        return [{"path": str(root), "size": -1, "mtime_ns": -1}]

    entries: list[dict[str, int | str]] = []
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        stat = path.stat()
        entries.append(
            {
                "path": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return entries


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False
