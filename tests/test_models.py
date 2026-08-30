from pathlib import Path

import pytest

from watermark_remover.models import PipelineConfig, QualityMetrics, Region


def test_region_clamps_to_frame():
    region = Region(-10, -5, 200, 100).clamp(100, 50)
    assert region == Region(0, 0, 100, 50)


def test_residual_fraction_handles_zero_support():
    metrics = QualityMetrics(0, 0, 0, 0.0, 0.0)
    assert metrics.residual_fraction == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("temporal_radius", -1),
        ("chunk_size", 0),
        ("scene_threshold", 1.1),
        ("min_scene_length", 0),
        ("alpha_inpaint_threshold", 1.1),
        ("analytic_confidence_min", -0.1),
        ("residual_dilate", -1),
        ("neighbor_length", 0),
        ("ref_stride", 0),
        ("resize_ratio", 0),
    ],
)
def test_pipeline_config_validation(field, value, tmp_path: Path):
    kwargs = dict(
        input_path=tmp_path/"in.mp4",
        propainter_dir=tmp_path/"p",
        output_path=tmp_path/"out.mp4",
        report_path=tmp_path/"out.csv",
    )
    kwargs[field] = value
    config = PipelineConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()
