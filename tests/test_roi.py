from pathlib import Path

import cv2
import numpy as np

from watermark_remover.models import Region
from watermark_remover.roi import crop_sequences, find_mask_union_roi


def test_find_mask_union_roi_adds_padding_and_clamps(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    first = np.zeros((80, 100), dtype=np.uint8)
    second = np.zeros((80, 100), dtype=np.uint8)
    first[10:20, 12:30] = 255
    second[15:25, 25:40] = 255
    cv2.imwrite(str(masks / "00000.png"), first)
    cv2.imwrite(str(masks / "00001.png"), second)

    roi = find_mask_union_roi(masks, 100, 80, padding=8)

    assert roi == Region(x=4, y=2, width=44, height=31)


def test_find_mask_union_roi_returns_none_for_empty_masks(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    cv2.imwrite(str(masks / "00000.png"), np.zeros((20, 30), dtype=np.uint8))
    assert find_mask_union_roi(masks, 30, 20, padding=4) is None


def test_crop_sequences_preserves_matching_names(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    masks = tmp_path / "masks"
    cropped_frames = tmp_path / "cropped-frames"
    cropped_masks = tmp_path / "cropped-masks"
    frames.mkdir()
    masks.mkdir()

    frame = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[3:7, 4:9] = 255
    cv2.imwrite(str(frames / "00000.png"), frame)
    cv2.imwrite(str(masks / "00000.png"), mask)

    crop_sequences(
        frames,
        masks,
        Region(x=3, y=2, width=7, height=6),
        cropped_frames,
        cropped_masks,
    )

    cropped_frame = cv2.imread(str(cropped_frames / "00000.png"))
    cropped_mask = cv2.imread(str(cropped_masks / "00000.png"), cv2.IMREAD_GRAYSCALE)
    assert cropped_frame is not None and cropped_frame.shape == (6, 7, 3)
    assert cropped_mask is not None and cropped_mask.shape == (6, 7)
