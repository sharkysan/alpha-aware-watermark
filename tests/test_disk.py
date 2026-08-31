from pathlib import Path
from types import SimpleNamespace

import pytest

from watermark_remover.disk import ensure_disk_space, estimate_disk_space


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


def test_estimate_disk_space_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        estimate_disk_space(
            width=0,
            height=1080,
            frame_count=100,
            input_size_bytes=1,
            save_debug=False,
        )
