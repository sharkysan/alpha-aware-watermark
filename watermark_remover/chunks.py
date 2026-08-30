from __future__ import annotations

from dataclasses import dataclass

from .scenes import Scene


@dataclass(frozen=True)
class FrameChunk:
    """Processing chunk plus the context range needed for temporal operations."""

    process_start: int
    process_end: int
    load_start: int
    load_end: int

    @property
    def process_length(self) -> int:
        return self.process_end - self.process_start

    @property
    def load_length(self) -> int:
        return self.load_end - self.load_start


def iter_scene_chunks(
    scene: Scene,
    chunk_size: int,
    temporal_radius: int,
) -> list[FrameChunk]:
    """Split a scene into bounded chunks with clipped temporal overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if temporal_radius < 0:
        raise ValueError("temporal_radius must be >= 0")

    chunks: list[FrameChunk] = []
    start = scene.start
    while start < scene.end:
        end = min(scene.end, start + chunk_size)
        chunks.append(
            FrameChunk(
                process_start=start,
                process_end=end,
                load_start=max(scene.start, start - temporal_radius),
                load_end=min(scene.end, end + temporal_radius),
            )
        )
        start = end
    return chunks
