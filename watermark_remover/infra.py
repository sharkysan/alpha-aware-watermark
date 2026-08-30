from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .ports import CommandRunner


class SubprocessCommandRunner:
    """Thin adapter around subprocess for dependency inversion and testability."""

    def run(self, args: list[str], cwd: Path | None = None) -> None:
        subprocess.run(args, cwd=cwd, check=True)


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


class ProPainterAdapter:
    """Adapter for the external ProPainter CLI."""

    def __init__(
        self,
        repo_dir: Path,
        runner: CommandRunner,
        neighbor_length: int,
        ref_stride: int,
        resize_ratio: float,
        fp16: bool,
    ) -> None:
        self.repo_dir = repo_dir
        self.runner = runner
        self.neighbor_length = neighbor_length
        self.ref_stride = ref_stride
        self.resize_ratio = resize_ratio
        self.fp16 = fp16

    @property
    def inference_script(self) -> Path:
        return self.repo_dir / "inference_propainter.py"

    def validate(self) -> None:
        if not self.inference_script.exists():
            raise FileNotFoundError(
                f"Could not find ProPainter inference script: {self.inference_script}"
            )

    def inpaint(self, frames_dir: Path, masks_dir: Path, output_dir: Path) -> Path:
        self.validate()
        output_dir.mkdir(parents=True, exist_ok=True)

        results_dir = self.repo_dir / "results"
        before = self._snapshot(results_dir)

        cmd = [
            sys.executable,
            str(self.inference_script),
            "--video",
            str(frames_dir),
            "--mask",
            str(masks_dir),
            "--neighbor_length",
            str(self.neighbor_length),
            "--ref_stride",
            str(self.ref_stride),
            "--resize_ratio",
            str(self.resize_ratio),
        ]
        if self.fp16:
            cmd.append("--fp16")

        self.runner.run(cmd, cwd=self.repo_dir)

        after = self._snapshot(results_dir)
        new_videos = sorted(
            [
                p
                for p in (after - before)
                if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
            ],
            key=lambda p: p.stat().st_mtime,
        )
        candidate = new_videos[-1] if new_videos else self._latest_video(results_dir)
        if candidate is None:
            raise RuntimeError("ProPainter finished but no output video was found.")

        copied = output_dir / candidate.name
        shutil.copy2(candidate, copied)
        return copied

    @staticmethod
    def _snapshot(root: Path) -> set[Path]:
        return set(root.rglob("*")) if root.exists() else set()

    @staticmethod
    def _latest_video(root: Path) -> Path | None:
        if not root.exists():
            return None
        videos = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
        ]
        return max(videos, key=lambda p: p.stat().st_mtime) if videos else None
