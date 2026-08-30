from pathlib import Path

import cv2
import numpy as np

from watermark_remover.mask_providers import DirectoryMaskProvider, RegionMaskProvider
from watermark_remover.models import Region


def test_region_provider_returns_expected_mask():
    provider = RegionMaskProvider(Region(2, 3, 4, 2), feather_sigma=0)
    mask = provider.get_mask(99, 10, 10)
    assert int((mask > 0).sum()) == 8


def test_directory_provider_prefers_frame_mask(tmp_path: Path):
    source = np.zeros((4, 4), dtype=np.uint8)
    source[1:3, 1:3] = 255
    cv2.imwrite(str(tmp_path / "00000.png"), source)

    fallback = RegionMaskProvider(Region(0, 0, 4, 4), feather_sigma=0)
    provider = DirectoryMaskProvider(tmp_path, fallback=fallback)

    mask = provider.get_mask(0, 4, 4)
    assert int((mask > 0).sum()) == 4


def test_directory_provider_uses_fallback_when_mask_missing(tmp_path: Path):
    fallback = RegionMaskProvider(Region(1, 1, 2, 2), feather_sigma=0)
    provider = DirectoryMaskProvider(tmp_path, fallback=fallback)

    mask = provider.get_mask(3, 4, 4)
    assert int((mask > 0).sum()) == 4
