from pathlib import Path

import pytest

from watermark_remover.checkpoints import CheckpointStore, build_checkpoint_fingerprint
from watermark_remover.chunks import FrameChunk
from watermark_remover.models import PipelineConfig, QualityMetrics, Region


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    input_path = tmp_path / "input.mp4"
    if not input_path.exists():
        input_path.write_bytes(b"video")
    values = {
        "input_path": input_path,
        "propainter_dir": tmp_path / "ProPainter",
        "output_path": tmp_path / "out.mp4",
        "report_path": tmp_path / "report.csv",
        "resume": True,
        "checkpoint_dir": tmp_path / "checkpoints",
    }
    values.update(overrides)
    return PipelineConfig(**values)


def _metrics(start: int, end: int) -> list[QualityMetrics]:
    return [
        QualityMetrics(
            frame_index=index,
            support_pixels=100,
            residual_pixels=10,
            mean_alpha=0.5,
            mean_confidence=0.8,
        )
        for index in range(start, end)
    ]


def test_fingerprint_changes_with_processing_settings(tmp_path: Path) -> None:
    base = _config(tmp_path)
    changed = _config(tmp_path, region=Region(10, 20, 30, 40))

    assert build_checkpoint_fingerprint(base) != build_checkpoint_fingerprint(changed)


def test_prepare_discards_incompatible_checkpoint_state(tmp_path: Path) -> None:
    original = CheckpointStore(_config(tmp_path))
    original.prepare()
    stale_file = original.analytic_dir / "00000.png"
    stale_file.write_bytes(b"old")

    changed = CheckpointStore(_config(tmp_path, temporal_radius=4))
    changed.prepare()

    assert not stale_file.exists()
    assert changed.manifest_path.is_file()


def test_complete_chunk_is_reused_only_with_all_outputs(tmp_path: Path) -> None:
    store = CheckpointStore(_config(tmp_path))
    store.prepare()
    chunk = FrameChunk(process_start=2, process_end=4, load_start=0, load_end=6)
    metrics = _metrics(2, 4)

    for index in range(2, 4):
        (store.analytic_dir / f"{index:05d}.png").write_bytes(b"analytic")
        (store.residual_dir / f"{index:05d}.png").write_bytes(b"mask")
    store.save_chunk(chunk, metrics)

    assert store.load_chunk(chunk) == metrics

    (store.residual_dir / "00003.png").unlink()
    assert store.load_chunk(chunk) is None


def test_zero_length_output_invalidates_chunk(tmp_path: Path) -> None:
    store = CheckpointStore(_config(tmp_path))
    store.prepare()
    chunk = FrameChunk(process_start=0, process_end=1, load_start=0, load_end=1)
    metrics = _metrics(0, 1)

    (store.analytic_dir / "00000.png").write_bytes(b"")
    (store.residual_dir / "00000.png").write_bytes(b"mask")
    store.save_chunk(chunk, metrics)

    assert store.load_chunk(chunk) is None


def test_cleanup_removes_checkpoint_directory(tmp_path: Path) -> None:
    store = CheckpointStore(_config(tmp_path))
    store.prepare()

    store.cleanup()

    assert not store.root.exists()


def test_checkpoint_directory_requires_resume(tmp_path: Path) -> None:
    config = PipelineConfig(
        input_path=tmp_path / "input.mp4",
        propainter_dir=tmp_path / "ProPainter",
        output_path=tmp_path / "out.mp4",
        report_path=tmp_path / "report.csv",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    with pytest.raises(ValueError, match="checkpoint_dir"):
        config.validate()
