from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PipelineProgress:
    """One monotonic progress update emitted by the processing pipeline."""

    fraction: float
    stage: str
    message: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("Progress fraction must be between 0 and 1")
        if not self.stage.strip():
            raise ValueError("Progress stage must not be empty")
        if not self.message.strip():
            raise ValueError("Progress message must not be empty")


class ProgressReporter(Protocol):
    def __call__(self, update: PipelineProgress) -> None:
        """Consume one pipeline progress update."""


def ignore_progress(update: PipelineProgress) -> None:
    """Default reporter used when callers do not need progress updates."""
    del update
