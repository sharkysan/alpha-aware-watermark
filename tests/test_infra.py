import subprocess
from pathlib import Path

import pytest

from watermark_remover.infra import ProPainterAdapter, SubprocessCommandRunner


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, args, cwd=None):
        self.calls.append((args, cwd))


def test_propainter_builds_expected_command(tmp_path: Path):
    repo = tmp_path / "ProPainter"
    repo.mkdir()
    (repo / "inference_propainter.py").write_text("# stub", encoding="utf-8")

    runner = FakeRunner()
    adapter = ProPainterAdapter(
        repo_dir=repo,
        runner=runner,
        neighbor_length=6,
        ref_stride=14,
        resize_ratio=0.75,
        fp16=True,
    )

    adapter.validate()
    assert adapter.inference_script.exists()


def test_subprocess_runner_surfaces_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=8,
            cmd=args[0],
            stderr="Unrecognized option 'vsync'.",
        )

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(RuntimeError) as exc_info:
        SubprocessCommandRunner().run(["ffmpeg", "-bad-option"])

    message = str(exc_info.value)
    assert "exit code 8" in message
    assert "ffmpeg" in message
    assert "Unrecognized option 'vsync'." in message


def test_subprocess_runner_captures_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def succeed(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", succeed)
    SubprocessCommandRunner().run(["ffmpeg", "-version"])

    _, kwargs = calls[0]
    assert kwargs["check"] is True
    assert kwargs["text"] is True
    assert kwargs["capture_output"] is True
