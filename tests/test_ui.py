from pathlib import Path

import numpy as np
import pytest

from watermark_remover.models import Region
from watermark_remover.progress import PipelineProgress
from watermark_remover.ui import (
    build_pipeline_config,
    process_video,
    region_from_annotation,
    region_from_points,
    update_region_selection,
)


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
        build_pipeline_config(tmp_path / "clip.mp4", tmp_path / "ProPainter", chunk_size=0)


def test_region_from_points_normalizes_direction_and_clamps() -> None:
    region = region_from_points((90, 70), (-5, 20), frame_width=100, frame_height=80)
    assert region == Region(x=0, y=20, width=91, height=51)


def test_region_from_annotation_maps_dragged_box() -> None:
    annotation = {
        "image": np.zeros((80, 100, 3), dtype=np.uint8),
        "boxes": [{"xmin": 12, "ymin": 17, "xmax": 52, "ymax": 44}],
    }
    assert region_from_annotation(annotation) == Region(x=12, y=17, width=40, height=27)


def test_region_from_annotation_normalizes_and_handles_empty_selection() -> None:
    assert region_from_annotation(None) is None
    assert region_from_annotation({"image": None, "boxes": []}) is None
    annotation = {
        "image": None,
        "boxes": [{"xmin": 50, "ymin": 30, "xmax": 10, "ymax": 5}],
    }
    assert region_from_annotation(annotation) == Region(x=10, y=5, width=40, height=25)


def test_update_region_selection_uses_two_opposite_corner_clicks() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    first = update_region_selection(frame, [], (10, 15))
    points = first[0]
    assert points == [[10, 15]]
    assert first[5] is True
    assert "opposite corner" in first[7]
    second = update_region_selection(frame, points, (40, 45))
    assert second[0] == [[10, 15], [40, 45]]
    assert second[1:5] == (10.0, 15.0, 31.0, 31.0)
    assert second[5] is True
    assert second[6].shape == frame.shape
    assert "width=31" in second[7]


def test_update_region_selection_restarts_after_completed_rectangle() -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    result = update_region_selection(frame, [[1, 2], [8, 9]], (12, 13))
    assert result[0] == [[12, 13]]
    assert result[1:5] == (12.0, 13.0, 1.0, 1.0)


def test_process_video_delegates_progress_and_returns_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_dir = tmp_path / "nested" / "results"
    calls = []
    updates: list[PipelineProgress] = []

    class FakePipeline:
        def __init__(self, config, progress_reporter=None):
            self.config = config
            self.progress_reporter = progress_reporter
            calls.append(config)

        def run(self):
            if self.progress_reporter is not None:
                self.progress_reporter(PipelineProgress(0.5, "processing", "Halfway"))
            self.config.report_path.write_text("frame_index\n", encoding="utf-8")
            self.config.output_path.write_bytes(b"video")
            return self.config.output_path

    monkeypatch.setattr("watermark_remover.ui.WatermarkRemovalPipeline", FakePipeline)
    video, report, status = process_video(
        tmp_path / "clip.mp4",
        tmp_path / "ProPainter",
        output_dir=output_dir,
        progress_reporter=updates.append,
    )
    assert output_dir.is_dir()
    assert Path(video) == output_dir / "clip_alpha_clean.mp4"
    assert Path(report) == output_dir / "clip_alpha_quality.csv"
    assert status == "Processing completed successfully."
    assert len(calls) == 1
    assert updates == [PipelineProgress(0.5, "processing", "Halfway")]
