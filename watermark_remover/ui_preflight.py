from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .disk import estimate_disk_space
from .video import probe_video


@dataclass(frozen=True)
class UiPreflightReport:
    ready: bool
    markdown: str


def build_ui_preflight_report(
    input_video: str | Path | None,
    propainter_dir: str | Path | None,
    output_dir: str | Path | None = None,
    *,
    save_debug: bool = False,
) -> UiPreflightReport:
    """Return a human-readable readiness report for one prospective UI run."""
    issues: list[str] = []
    rows: list[tuple[str, str]] = []

    input_path = Path(input_video).expanduser() if input_video else None
    if input_path is None or not input_path.exists():
        issues.append("Input video is missing.")
    else:
        width, height, fps, frame_count = probe_video(input_path)
        rows.extend(
            [
                ("Video", f"{width}×{height} · {fps:.2f} FPS · {frame_count} frames"),
                ("Input size", _format_bytes(input_path.stat().st_size)),
            ]
        )

        destination = (
            Path(output_dir).expanduser()
            if output_dir is not None and str(output_dir).strip()
            else input_path.parent
        )
        estimate = estimate_disk_space(
            width=width,
            height=height,
            frame_count=frame_count,
            input_size_bytes=input_path.stat().st_size,
            save_debug=save_debug,
        )
        scratch_free = shutil.disk_usage(_nearest_existing_path(Path(tempfile.gettempdir()))).free
        output_free = shutil.disk_usage(_nearest_existing_path(destination)).free
        rows.extend(
            [
                ("Estimated scratch", _format_bytes(estimate.scratch_bytes)),
                ("Scratch available", _format_bytes(scratch_free)),
                ("Estimated output", _format_bytes(estimate.output_bytes)),
                ("Output available", _format_bytes(output_free)),
            ]
        )
        if scratch_free < estimate.scratch_bytes:
            issues.append("Scratch filesystem does not have enough free space.")
        if output_free < estimate.output_bytes:
            issues.append("Output filesystem does not have enough free space.")

    ffmpeg_path = shutil.which("ffmpeg")
    rows.append(("FFmpeg", ffmpeg_path or "Not found"))
    if ffmpeg_path is None:
        issues.append("FFmpeg is not available on PATH.")

    painter_path = Path(propainter_dir).expanduser() if propainter_dir else None
    painter_ready = painter_path is not None and painter_path.is_dir()
    rows.append(("ProPainter", str(painter_path) if painter_ready else "Directory not found"))
    if not painter_ready:
        issues.append("ProPainter directory is missing or invalid.")

    ready = not issues
    title = "✅ Ready to process" if ready else "⚠️ Preflight needs attention"
    lines = [f"### {title}", "", "| Check | Result |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    if issues:
        lines.extend(["", "**Issues**", *[f"- {issue}" for issue in issues]])
    return UiPreflightReport(ready=ready, markdown="\n".join(lines))


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")
