from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import Region


def region_mask(
    frame_width: int,
    frame_height: int,
    region: Region,
    feather_sigma: float = 1.0,
) -> np.ndarray:
    region = region.clamp(frame_width, frame_height)
    mask = np.zeros((frame_height, frame_width), np.uint8)
    x2 = region.x + region.width
    y2 = region.y + region.height
    mask[region.y:y2, region.x:x2] = 255

    if feather_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather_sigma)

    return mask


def load_frame_mask(
    mask_dir: Path,
    frame_index: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray | None:
    candidates = [
        mask_dir / f"{frame_index:06d}.png",
        mask_dir / f"{frame_index:05d}.png",
        mask_dir / f"{frame_index:06d}.jpg",
        mask_dir / f"{frame_index:05d}.jpg",
    ]
    for path in candidates:
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (frame_height, frame_width):
            mask = cv2.resize(
                mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask

    return None
