from pathlib import Path

import numpy as np

from watermark_remover.models import PipelineConfig
from watermark_remover.pipeline import WatermarkRemovalPipeline
from watermark_remover.scenes import Scene


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    values = dict(
        input_path=tmp_path / "input.mp4",
        propainter_dir=tmp_path / "ProPainter",
        output_path=tmp_path / "out.mp4",
        report_path=tmp_path / "report.csv",
        temporal_radius=2,
    )
    values.update(overrides)
    return PipelineConfig(**values)


def test_neighbor_indices_are_clipped_to_scene(tmp_path: Path):
    pipeline = WatermarkRemovalPipeline(_config(tmp_path))
    scene = Scene(10, 20)

    assert pipeline._neighbor_indices(10, scene) == [10, 11, 12]
    assert pipeline._neighbor_indices(19, scene) == [17, 18, 19]
    assert pipeline._neighbor_indices(15, scene) == [13, 14, 15, 16, 17]


def test_background_estimation_without_motion_uses_temporal_median(tmp_path: Path):
    pipeline = WatermarkRemovalPipeline(_config(tmp_path, motion_compensation=False))
    observed = np.full((2, 2, 3), 50, dtype=np.uint8)
    neighbors = [
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 90, dtype=np.uint8),
    ]

    result = pipeline._estimate_background(observed, neighbors)
    assert np.all(result == 50)
