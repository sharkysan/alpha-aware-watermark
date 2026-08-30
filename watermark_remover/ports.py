from __future__ import annotations

from pathlib import Path
from typing import Protocol


class CommandRunner(Protocol):
    def run(self, args: list[str], cwd: Path | None = None) -> None:
        """Run an external command or raise on failure."""


class VideoInpainter(Protocol):
    def inpaint(self, frames_dir: Path, masks_dir: Path, output_dir: Path) -> Path:
        """Return the generated video path."""
