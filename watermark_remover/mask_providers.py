from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .masks import load_frame_mask, region_mask
from .models import Region


class MaskProvider(Protocol):
    """Strategy for providing a watermark support mask for a video frame."""

    def get_mask(
        self,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Return an uint8 support mask in frame coordinates."""


@dataclass(frozen=True)
class RegionMaskProvider:
    """Provide the same rectangular support region for every frame."""

    region: Region
    feather_sigma: float = 1.0

    def get_mask(
        self,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        del frame_index
        return region_mask(
            frame_width,
            frame_height,
            self.region,
            feather_sigma=self.feather_sigma,
        )


@dataclass(frozen=True)
class DirectoryMaskProvider:
    """Read per-frame masks from a directory with an optional fallback strategy."""

    mask_dir: Path
    fallback: MaskProvider | None = None

    def get_mask(
        self,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        mask = load_frame_mask(
            self.mask_dir,
            frame_index,
            frame_width,
            frame_height,
        )
        if mask is not None:
            return mask
        if self.fallback is not None:
            return self.fallback.get_mask(
                frame_index,
                frame_width,
                frame_height,
            )
        raise FileNotFoundError(
            f"No mask found for frame {frame_index} in {self.mask_dir}"
        )
