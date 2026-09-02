from pathlib import Path

from watermark_remover.video import extract_frames


class ExtractRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, args: list[str], cwd: Path | None = None) -> None:
        self.calls.append(args)
        pattern = Path(args[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)
        (pattern.parent / "000001.png").write_bytes(b"png")
        (pattern.parent / "000002.png").write_bytes(b"png")


def test_extract_frames_uses_ffmpeg9_compatible_arguments(tmp_path: Path) -> None:
    runner = ExtractRunner()
    input_path = tmp_path / "input.mp4"
    output_dir = tmp_path / "frames"

    frames = extract_frames(input_path, output_dir, runner)

    assert len(frames) == 2
    command = runner.calls[0]
    assert command[:4] == ["ffmpeg", "-y", "-i", str(input_path)]
    assert "-vsync" not in command
    assert command[-1] == str(output_dir / "%06d.png")
