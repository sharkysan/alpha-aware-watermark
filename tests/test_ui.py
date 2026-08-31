from pathlib import Path

import pytest

from watermark_remover.models import Region
from watermark_remover.ui import build_pipeline_config, process_video


def test_build_pipeline_config_uses_input_directory_by_default(tmp_path: Path):
    input_path = tmp_path / "clip.mp4"

    config = build_pipeline_config(input_path, tmp_path / "ProPainter")

    assert config.input_path == input_path
    assert config.output_path == tmp_path / "clip_alpha_clean.mp4"
    assert config.report_path == tmp_path / "clip_alpha_quality.csv"
    assert config.region is None
    assert config.mask_dir is None


def test_build_pipeline_config_maps_custom_region_and_mask(tmp_path: Path):
    output_dir = tmp_path / "results"
    mask_dir = tmp_path / "masks"

    config = build_pipeline_config(
        tmp_path / "clip.mp4",
        tmp_path / "ProPainter",
        output_dir=output_dir,
        mask_dir=mask_dir,
        custom_region=True,
        x=11,
        y=22,
        width=123,
        height=45,
        temporal_radius=4,
        chunk_size=48,
        motion_compensation=False,
        fp16=True,
    )

    assert config.output_path == output_dir / "clip_alpha_clean.mp4"
    assert config.report_path == output_dir / "clip_alpha_quality.csv"
    assert config.mask_dir == mask_dir
    assert config.region == Region(11, 22, 123, 45)
    assert config.temporal_radius == 4
    assert config.chunk_size == 48
    assert config.motion_compensation is False
    assert config.fp16 is True


def test_build_pipeline_config_reuses_model_validation(tmp_path: Path):
    with pytest.raises(ValueError, match="chunk_size"):
        build_pipeline_config(
            tmp_path / "clip.mp4",
            tmp_path / "ProPainter",
            chunk_size=0,
        )


def test_process_video_creates_destination_and_returns_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_dir = tmp_path / "nested" / "results"
    calls = []

    class FakePipeline:
        def __init__(self, config):
            self.config = config
            calls.append(config)

        def run(self):
            self.config.report_path.write_text("frame_index\n", encoding="utf-8")
            self.config.output_path.write_bytes(b"video")
            return self.config.output_path

    monkeypatch.setattr("watermark_remover.ui.WatermarkRemovalPipeline", FakePipeline)

    video, report, status = process_video(
        tmp_path / "clip.mp4",
        tmp_path / "ProPainter",
        output_dir=output_dir,
    )

    assert output_dir.is_dir()
    assert Path(video) == output_dir / "clip_alpha_clean.mp4"
    assert Path(report) == output_dir / "clip_alpha_quality.csv"
    assert status == "Processing completed successfully."
    assert len(calls) == 1
