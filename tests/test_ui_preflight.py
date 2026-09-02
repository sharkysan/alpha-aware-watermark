import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from watermark_remover.ui_preflight import build_ui_preflight_report


def _successful_python_probe(*args, **kwargs):
    return subprocess.CompletedProcess(
        args=args[0],
        returncode=0,
        stdout="3.11.9\n2.7.1\n",
        stderr="",
    )


def test_ui_preflight_reports_ready_video_and_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    propainter = tmp_path / "ProPainter"
    propainter.mkdir()
    (propainter / "inference_propainter.py").write_text("# stub", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")

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
    monkeypatch.setattr(
        "watermark_remover.ui_preflight.subprocess.run",
        _successful_python_probe,
    )

    report = build_ui_preflight_report(
        video,
        propainter,
        tmp_path,
        propainter_python=python,
    )

    assert report.ready is True
    assert "Ready to process" in report.markdown
    assert "1920×1080" in report.markdown
    assert "Estimated scratch" in report.markdown
    assert "/usr/bin/ffmpeg" in report.markdown
    assert str(python) in report.markdown
    assert "imports OK" in report.markdown


def test_ui_preflight_reports_missing_propainter_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    propainter = tmp_path / "ProPainter"
    propainter.mkdir()
    (propainter / "inference_propainter.py").write_text("# stub", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")

    monkeypatch.setattr(
        "watermark_remover.ui_preflight.probe_video",
        lambda _: (1920, 1080, 30.0, 100),
    )
    monkeypatch.setattr(
        "watermark_remover.ui_preflight.shutil.disk_usage",
        lambda _: SimpleNamespace(free=10 * 1024**3),
    )
    monkeypatch.setattr(
        "watermark_remover.ui_preflight.shutil.which",
        lambda _: "/usr/bin/ffmpeg",
    )

    def failed_probe(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr="ModuleNotFoundError: No module named 'imageio'",
        )

    monkeypatch.setattr("watermark_remover.ui_preflight.subprocess.run", failed_probe)

    report = build_ui_preflight_report(
        video,
        propainter,
        tmp_path,
        propainter_python=python,
    )

    assert report.ready is False
    assert "Import check failed" in report.markdown
    assert "imageio" in report.markdown


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

    report = build_ui_preflight_report(
        video,
        tmp_path / "missing",
        tmp_path,
        propainter_python=tmp_path / "missing-python.exe",
    )

    assert report.ready is False
    assert "Preflight needs attention" in report.markdown
    assert "does not have enough free space" in report.markdown
    assert "FFmpeg is not available" in report.markdown
    assert "ProPainter directory is missing" in report.markdown
    assert "Python executable is missing" in report.markdown
