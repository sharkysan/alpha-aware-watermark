from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    propainter_python: str | Path | None = None,
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
        scratch_probe = _nearest_existing_path(Path(tempfile.gettempdir()))
        output_probe = _nearest_existing_path(destination)
        scratch_free = shutil.disk_usage(scratch_probe).free
        output_free = shutil.disk_usage(output_probe).free
        same_filesystem = os.stat(scratch_probe).st_dev == os.stat(output_probe).st_dev
        rows.extend(
            [
                ("Estimated scratch", _format_bytes(estimate.scratch_bytes)),
                ("Estimated output", _format_bytes(estimate.output_bytes)),
            ]
        )
        if same_filesystem:
            combined_required = estimate.scratch_bytes + estimate.output_bytes
            rows.append(("Shared filesystem available", _format_bytes(scratch_free)))
            if scratch_free < combined_required:
                issues.append(
                    "Shared scratch/output filesystem does not have enough free space."
                )
        else:
            rows.extend(
                [
                    ("Scratch available", _format_bytes(scratch_free)),
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
    elif not (painter_path / "inference_propainter.py").is_file():
        issues.append("ProPainter inference_propainter.py was not found in the selected directory.")

    python_path = _resolve_python(propainter_python)
    rows.append(("ProPainter Python", str(python_path)))
    if not python_path.is_file():
        issues.append("ProPainter Python executable is missing or invalid.")
    elif painter_ready:
        environment_result = _check_propainter_environment(python_path, painter_path)
        rows.append(("ProPainter environment", environment_result[0]))
        if environment_result[1] is not None:
            issues.append(environment_result[1])

    ready = not issues
    title = "✅ Ready to process" if ready else "⚠️ Preflight needs attention"
    lines = [f"### {title}", "", "| Check | Result |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    if issues:
        lines.extend(["", "**Issues**", *[f"- {issue}" for issue in issues]])
    return UiPreflightReport(ready=ready, markdown="\n".join(lines))


def _resolve_python(value: str | Path | None) -> Path:
    if value is not None and str(value).strip():
        return Path(value).expanduser()
    return Path(sys.executable)


def _check_propainter_environment(
    python_path: Path,
    propainter_dir: Path,
) -> tuple[str, str | None]:
    probe = (
        "import sys, imageio, torch, torchvision, cv2, numpy; "
        "print(sys.version.split()[0]); print(torch.__version__)"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe],
            cwd=propainter_dir,
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)
        detail = str(stderr).strip().splitlines()[-1] if str(stderr).strip() else str(exc)
        return "Import check failed", f"ProPainter Python environment is not ready: {detail}"

    values = result.stdout.strip().splitlines()
    python_version = values[0] if values else "unknown"
    torch_version = values[1] if len(values) > 1 else "unknown"
    return f"Python {python_version} · torch {torch_version} · imports OK", None


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
