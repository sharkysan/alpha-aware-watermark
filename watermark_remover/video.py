from __future__ import annotations

from pathlib import Path

import cv2

from .ports import CommandRunner


def probe_video(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return width, height, fps, frame_count


def extract_frames(
    input_path: Path,
    output_dir: Path,
    runner: CommandRunner,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            str(output_dir / "%06d.png"),
        ]
    )
    frames = sorted(output_dir.glob("*.png"))
    if not frames:
        raise RuntimeError("FFmpeg extracted no frames.")
    return frames


def mux_original_audio(
    generated_video: Path,
    original_video: Path,
    output_path: Path,
    runner: CommandRunner,
) -> None:
    runner.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(generated_video),
            "-i",
            str(original_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
