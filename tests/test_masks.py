from pathlib import Path

import cv2
import numpy as np

from watermark_remover.masks import load_frame_mask, region_mask
from watermark_remover.models import Region


def test_region_mask_has_expected_support():
    mask = region_mask(10, 10, Region(2, 3, 4, 2), feather_sigma=0)
    assert int((mask > 0).sum()) == 8


def test_load_frame_mask_resizes(tmp_path: Path):
    source = np.full((2,2), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path/"00000.png"), source)

    mask = load_frame_mask(tmp_path, 0, 4, 4)

    assert mask is not None
    assert mask.shape == (4,4)
    assert np.all(mask == 255)
