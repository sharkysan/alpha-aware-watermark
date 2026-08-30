from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Scene:
    """Half-open frame range [start, end) for one visual shot."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("scene start must be >= 0")
        if self.end <= self.start:
            raise ValueError("scene end must be greater than start")

    @property
    def length(self) -> int:
        return self.end - self.start


def _hsv_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def detect_scenes(
    frame_paths: list[Path],
    threshold: float = 0.62,
    min_scene_length: int = 10,
) -> list[Scene]:
    """Detect hard scene cuts using HSV histogram correlation."""
    if not frame_paths:
        return []
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("scene threshold must be in [-1, 1]")
    if min_scene_length <= 0:
        raise ValueError("min_scene_length must be > 0")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")

    cuts = [0]
    previous_hist = _hsv_histogram(first)
    last_cut = 0

    for index, path in enumerate(frame_paths[1:], start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {path}")
        current_hist = _hsv_histogram(frame)
        similarity = cv2.compareHist(
            previous_hist,
            current_hist,
            cv2.HISTCMP_CORREL,
        )
        if similarity < threshold and index - last_cut >= min_scene_length:
            cuts.append(index)
            last_cut = index
        previous_hist = current_hist

    cuts.append(len(frame_paths))
    return [Scene(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
