import pytest

from watermark_remover.progress import PipelineProgress


def test_pipeline_progress_accepts_bounds() -> None:
    assert PipelineProgress(0.0, "start", "Starting").fraction == 0.0
    assert PipelineProgress(1.0, "complete", "Done").fraction == 1.0


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_pipeline_progress_rejects_out_of_range_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        PipelineProgress(fraction, "stage", "message")


def test_pipeline_progress_requires_stage_and_message() -> None:
    with pytest.raises(ValueError, match="stage"):
        PipelineProgress(0.5, " ", "message")
    with pytest.raises(ValueError, match="message"):
        PipelineProgress(0.5, "stage", " ")
