from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    """Axis-aligned watermark support region."""

    x: int
    y: int
    width: int
    height: int

    def clamp(self, frame_width: int, frame_height: int) -> "Region":
        x = max(0, min(self.x, frame_width - 1))
        y = max(0, min(self.y, frame_height - 1))
        width = max(1, min(self.width, frame_width - x))
        height = max(1, min(self.height, frame_height - y))
        return Region(x, y, width, height)


@dataclass(frozen=True)
class QualityMetrics:
    frame_index: int
    support_pixels: int
    residual_pixels: int
    mean_alpha: float
    mean_confidence: float

    @property
    def residual_fraction(self) -> float:
        return self.residual_pixels / max(1, self.support_pixels)


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    propainter_dir: Path
    output_path: Path
    report_path: Path
    mask_dir: Path | None = None
    region: Region | None = None
    temporal_radius: int = 2
    chunk_size: int = 24
    scene_threshold: float = 0.62
    min_scene_length: int = 10
    motion_compensation: bool = True
    alpha_inpaint_threshold: float = 0.55
    analytic_confidence_min: float = 0.28
    residual_dilate: int = 3
    neighbor_length: int = 10
    ref_stride: int = 10
    resize_ratio: float = 1.0
    fp16: bool = False
    save_debug: bool = False

    def validate(self) -> None:
        if self.temporal_radius < 0:
            raise ValueError("temporal_radius must be >= 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not -1.0 <= self.scene_threshold <= 1.0:
            raise ValueError("scene_threshold must be in [-1, 1]")
        if self.min_scene_length <= 0:
            raise ValueError("min_scene_length must be > 0")
        if not 0.0 <= self.alpha_inpaint_threshold <= 1.0:
            raise ValueError("alpha_inpaint_threshold must be in [0, 1]")
        if not 0.0 <= self.analytic_confidence_min <= 1.0:
            raise ValueError("analytic_confidence_min must be in [0, 1]")
        if self.residual_dilate < 0:
            raise ValueError("residual_dilate must be >= 0")
        if self.neighbor_length <= 0:
            raise ValueError("neighbor_length must be > 0")
        if self.ref_stride <= 0:
            raise ValueError("ref_stride must be > 0")
        if self.resize_ratio <= 0:
            raise ValueError("resize_ratio must be > 0")
