from pathlib import Path
from types import SimpleNamespace

import pytest

from watermark_remover.disk import (
    DiskSpaceEstimate,
    ensure_disk_space,
    ensure_pipeline_disk_space,
    estimate_disk_space,
)


def test_estimate_disk_space_accounts_for_intermediate_frames() -> None:
    estimate = estimate_disk_space(
        width=100,
        height=50,
        frame_count=10,
        input_size_bytes=20 * 1024 * 1024,
        save_debug=False,
    )

    assert estimate.scratch_bytes == 625_000
    assert estimate.output_bytes == 64 * 1024 * 1024


def test_estimate_disk_space_adds_debug_frame_budget() -> None:
    normal = estimate_disk_space(
        width=100,
        height=50,
        frame_count=10,
        input_size_bytes=100 * 1024 * 1024,
        save_debug=False,
    )
    debug = estimate_disk_space(
        width=100,
        height=50,
        frame_count=10,
        input_size_bytes=100 * 1024 * 1024,
        save_debug=True,
    )

    assert debug.scratch_bytes > normal.scratch_bytes
    assert debug.output_bytes == 200 * 1024 * 1024


def test_ensure_disk_space_accepts_sufficient_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "watermark_remover.disk.shutil.disk_usage",
        lambda _: SimpleNamespace(free=1_000),
    )

    ensure_disk_space(tmp_path, 999, label="scratch")


def test_ensure_disk_space_reports_required_and_available_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "watermark_remover.disk.shutil.disk_usage",
        lambda _: SimpleNamespace(free=512),
    )

    with pytest.raises(RuntimeError, match="Insufficient output disk space") as error:
        ensure_disk_space(tmp_path / "not-created-yet", 1024, label="output")

    assert "1.0 KiB" in str(error.value)
    assert "512.0 B" in str(error.value)


def test_pipeline_disk_space_combines_requirements_on_same_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    scratch.mkdir()
    output.mkdir()
    monkeypatch.setattr("watermark_remover.disk._filesystem_device", lambda _: 7)
    monkeypatch.setattr(
        "watermark_remover.disk.shutil.disk_usage",
        lambda _: SimpleNamespace(free=1_500),
    )

    with pytest.raises(RuntimeError, match="combined scratch/output"):
        ensure_pipeline_disk_space(
            scratch,
            output,
            DiskSpaceEstimate(scratch_bytes=900, output_bytes=700),
        )


def test_pipeline_disk_space_checks_distinct_filesystems_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    scratch.mkdir()
    output.mkdir()

    def device(path: Path) -> int:
        return 1 if path == scratch else 2

    def disk_usage(path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=1_000 if path == scratch else 800)

    monkeypatch.setattr("watermark_remover.disk._filesystem_device", device)
    monkeypatch.setattr("watermark_remover.disk.shutil.disk_usage", disk_usage)

    ensure_pipeline_disk_space(
        scratch,
        output,
        DiskSpaceEstimate(scratch_bytes=900, output_bytes=700),
    )


def test_estimate_disk_space_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        estimate_disk_space(
            width=0,
            height=1080,
            frame_count=100,
            input_size_bytes=1,
            save_debug=False,
        )
