from pathlib import Path
from types import SimpleNamespace

import pytest

from watermark_remover.ui_preflight import build_ui_preflight_report


def test_ui_preflight_reports_ready_video_and_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    propainter = tmp_path / "ProPainter"
    propainter.mkdir()

    monkeypatch.setattr(
        "watermark_remover.ui_preflight.probe_video",
        lambda _: (1920, 1080, 29.97, 100),
    )
    monkeypatch.setattr(
        "watermark_remover.ui_preflight.shutil.disk_usage",
        lambda _: SimpleNamespace(free=10 * 1024**3),
    )
    monkeypatch.setattr(
        "watermark_remover.ui_preflight.shutil.which",
        lambda _: "/usr/bin/ffmpeg",
    )

    report = build_ui_preflight_report(video, propainter, tmp_path)

    assert report.ready is True
    assert "Ready to process" in report.markdown
    assert "1920×1080" in report.markdown
    assert "Estimated scratch" in report.markdown
    assert "/usr/bin/ffmpeg" in report.markdown


def test_ui_preflight_reports_actionable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    monkeypatch.setattr(
        "watermark_remover.ui_preflight.probe_video",
        lambda _: (3840, 2160, 30.0, 10_000),
    )
    monkeypatch.setattr(
        "watermark_remover.ui_preflight.shutil.disk_usage",
        lambda _: SimpleNamespace(free=1024),
    )
    monkeypatch.setattr(
        "watermark_remover.ui_preflight.shutil.which",
        lambda _: None,
    )

    report = build_ui_preflight_report(video, tmp_path / "missing", tmp_path)

    assert report.ready is False
    assert "Preflight needs attention" in report.markdown
    assert "does not have enough free space" in report.markdown
    assert "FFmpeg is not available" in report.markdown
    assert "ProPainter directory is missing" in report.markdown
