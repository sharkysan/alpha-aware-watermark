from pathlib import Path

import cv2
import numpy as np

from watermark_remover.scenes import Scene, detect_scenes


def _write_frame(path: Path, bgr: tuple[int, int, int]) -> None:
    frame = np.full((32, 32, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def test_detect_scenes_splits_hard_color_change(tmp_path: Path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index:05d}.png"
        _write_frame(path, (0, 0, 255))
        paths.append(path)
    for index in range(3, 6):
        path = tmp_path / f"{index:05d}.png"
        _write_frame(path, (255, 0, 0))
        paths.append(path)

    scenes = detect_scenes(paths, threshold=0.9, min_scene_length=2)
    assert scenes == [Scene(0, 3), Scene(3, 6)]


def test_empty_input_has_no_scenes():
    assert detect_scenes([]) == []
