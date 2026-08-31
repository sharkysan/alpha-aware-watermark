from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import PipelineConfig, Region
from .pipeline import WatermarkRemovalPipeline
from .progress import PipelineProgress, ProgressReporter
from .ui_preflight import build_ui_preflight_report


def build_pipeline_config(
    input_video: str | Path,
    propainter_dir: str | Path,
    output_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    custom_region: bool = False,
    x: float = 0,
    y: float = 0,
    width: float = 1,
    height: float = 1,
    temporal_radius: float = 2,
    chunk_size: float = 24,
    scene_threshold: float = 0.62,
    min_scene_length: float = 10,
    motion_compensation: bool = True,
    alpha_inpaint_threshold: float = 0.55,
    analytic_confidence_min: float = 0.28,
    residual_dilate: float = 3,
    neighbor_length: float = 10,
    ref_stride: float = 10,
    resize_ratio: float = 1.0,
    fp16: bool = False,
    save_debug: bool = False,
) -> PipelineConfig:
    """Translate UI values into the same PipelineConfig used by the CLI."""
    input_path = Path(input_video).expanduser()
    propainter_path = Path(propainter_dir).expanduser()
    destination = (
        Path(output_dir).expanduser()
        if output_dir is not None and str(output_dir).strip()
        else input_path.parent
    )
    output_path = destination / f"{input_path.stem}_alpha_clean.mp4"
    report_path = destination / f"{input_path.stem}_alpha_quality.csv"

    resolved_mask_dir = None
    if mask_dir is not None and str(mask_dir).strip():
        resolved_mask_dir = Path(mask_dir).expanduser()

    region = None
    if custom_region:
        region = Region(x=int(x), y=int(y), width=int(width), height=int(height))

    config = PipelineConfig(
        input_path=input_path,
        propainter_dir=propainter_path,
        output_path=output_path,
        report_path=report_path,
        mask_dir=resolved_mask_dir,
        region=region,
        temporal_radius=int(temporal_radius),
        chunk_size=int(chunk_size),
        scene_threshold=float(scene_threshold),
        min_scene_length=int(min_scene_length),
        motion_compensation=motion_compensation,
        alpha_inpaint_threshold=float(alpha_inpaint_threshold),
        analytic_confidence_min=float(analytic_confidence_min),
        residual_dilate=int(residual_dilate),
        neighbor_length=int(neighbor_length),
        ref_stride=int(ref_stride),
        resize_ratio=float(resize_ratio),
        fp16=fp16,
        save_debug=save_debug,
    )
    config.validate()
    return config


def extract_preview_frame(input_video: str | Path | None) -> np.ndarray | None:
    """Read the first video frame as RGB for interactive region selection."""
    if input_video is None or not str(input_video).strip():
        return None
    capture = cv2.VideoCapture(str(input_video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a preview frame from {input_video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def region_from_points(
    first: tuple[int, int],
    second: tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> Region:
    """Build a clamped positive rectangle from two opposite-corner clicks."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")
    x1 = min(max(first[0], 0), frame_width - 1)
    y1 = min(max(first[1], 0), frame_height - 1)
    x2 = min(max(second[0], 0), frame_width - 1)
    y2 = min(max(second[1], 0), frame_height - 1)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return Region(
        x=left,
        y=top,
        width=max(1, right - left + 1),
        height=max(1, bottom - top + 1),
    )


def update_region_selection(
    frame: np.ndarray | None,
    points: list[list[int]] | None,
    point: tuple[int, int],
) -> tuple[list[list[int]], float, float, float, float, bool, np.ndarray, str]:
    """Apply one preview click and return updated region controls/overlay."""
    if frame is None:
        raise ValueError("Upload a video before selecting a watermark region")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Preview frame must be an RGB image")
    frame_height, frame_width = frame.shape[:2]
    px = min(max(int(point[0]), 0), frame_width - 1)
    py = min(max(int(point[1]), 0), frame_height - 1)
    current = [] if not points or len(points) >= 2 else [list(points[0])]
    current.append([px, py])
    overlay = frame.copy()
    if len(current) == 1:
        cv2.drawMarker(
            overlay,
            (px, py),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        return (
            current,
            float(px),
            float(py),
            1.0,
            1.0,
            True,
            overlay,
            "First corner selected. Click the opposite corner.",
        )
    region = region_from_points(
        (current[0][0], current[0][1]),
        (current[1][0], current[1][1]),
        frame_width,
        frame_height,
    )
    cv2.rectangle(
        overlay,
        (region.x, region.y),
        (region.x + region.width - 1, region.y + region.height - 1),
        (255, 255, 255),
        2,
    )
    status = (
        f"Selected region: x={region.x}, y={region.y}, "
        f"width={region.width}, height={region.height}."
    )
    return (
        current,
        float(region.x),
        float(region.y),
        float(region.width),
        float(region.height),
        True,
        overlay,
        status,
    )


def run_ui_preflight(
    input_video: str | Path | None,
    propainter_dir: str | Path | None,
    output_dir: str | Path | None,
    save_debug: bool,
) -> str:
    """Return the rendered preflight report used by the Gradio panel."""
    return build_ui_preflight_report(
        input_video,
        propainter_dir,
        output_dir,
        save_debug=save_debug,
    ).markdown


def process_video(
    input_video: str | Path,
    propainter_dir: str | Path,
    output_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    custom_region: bool = False,
    x: float = 0,
    y: float = 0,
    width: float = 1,
    height: float = 1,
    temporal_radius: float = 2,
    chunk_size: float = 24,
    scene_threshold: float = 0.62,
    min_scene_length: float = 10,
    motion_compensation: bool = True,
    alpha_inpaint_threshold: float = 0.55,
    analytic_confidence_min: float = 0.28,
    residual_dilate: float = 3,
    neighbor_length: float = 10,
    ref_stride: float = 10,
    resize_ratio: float = 1.0,
    fp16: bool = False,
    save_debug: bool = False,
    progress_reporter: ProgressReporter | None = None,
) -> tuple[str, str, str]:
    """Run one UI request and return video path, report path, and status text."""
    config = build_pipeline_config(
        input_video=input_video,
        propainter_dir=propainter_dir,
        output_dir=output_dir,
        mask_dir=mask_dir,
        custom_region=custom_region,
        x=x,
        y=y,
        width=width,
        height=height,
        temporal_radius=temporal_radius,
        chunk_size=chunk_size,
        scene_threshold=scene_threshold,
        min_scene_length=min_scene_length,
        motion_compensation=motion_compensation,
        alpha_inpaint_threshold=alpha_inpaint_threshold,
        analytic_confidence_min=analytic_confidence_min,
        residual_dilate=residual_dilate,
        neighbor_length=neighbor_length,
        ref_stride=ref_stride,
        resize_ratio=resize_ratio,
        fp16=fp16,
        save_debug=save_debug,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    result = WatermarkRemovalPipeline(
        config,
        progress_reporter=progress_reporter,
    ).run()
    return str(result), str(config.report_path), "Processing completed successfully."


def build_app() -> Any:
    """Build the optional local Gradio application."""
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            'Gradio UI dependencies are not installed. Run: pip install -e ".[ui]"'
        ) from exc

    def load_preview(video: str | None) -> tuple[np.ndarray | None, list[list[int]], str]:
        return extract_preview_frame(video), [], "Click two opposite corners of the watermark."

    def select_region(
        frame: np.ndarray | None,
        points: list[list[int]] | None,
        evt: Any,
    ) -> tuple[list[list[int]], float, float, float, float, bool, np.ndarray, str]:
        index = evt.index
        if not isinstance(index, (tuple, list)) or len(index) < 2:
            raise ValueError("Could not determine the selected image coordinates")
        return update_region_selection(frame, points, (int(index[0]), int(index[1])))

    def process_with_progress(
        input_video: str | Path,
        propainter_dir: str | Path,
        output_dir: str | Path | None,
        mask_dir: str | Path | None,
        custom_region: bool,
        x: float,
        y: float,
        width: float,
        height: float,
        temporal_radius: float,
        chunk_size: float,
        scene_threshold: float,
        min_scene_length: float,
        motion_compensation: bool,
        alpha_inpaint_threshold: float,
        analytic_confidence_min: float,
        residual_dilate: float,
        neighbor_length: float,
        ref_stride: float,
        resize_ratio: float,
        fp16: bool,
        save_debug: bool,
        progress: Any = gr.Progress(),
    ) -> tuple[str, str, str]:
        def report(update: PipelineProgress) -> None:
            progress(update.fraction, desc=update.message)

        return process_video(
            input_video,
            propainter_dir,
            output_dir,
            mask_dir,
            custom_region,
            x,
            y,
            width,
            height,
            temporal_radius,
            chunk_size,
            scene_threshold,
            min_scene_length,
            motion_compensation,
            alpha_inpaint_threshold,
            analytic_confidence_min,
            residual_dilate,
            neighbor_length,
            ref_stride,
            resize_ratio,
            fp16,
            save_debug,
            progress_reporter=report,
        )

    with gr.Blocks(title="Alpha-Aware Watermark Remover") as app:
        gr.Markdown(
            "# Alpha-Aware Watermark Remover\n"
            "Local UI for the existing scene-aware, temporal processing pipeline."
        )
        region_points = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="Input video", sources=["upload"])
                preview_image = gr.Image(
                    label="Watermark region selector — click two opposite corners",
                    type="numpy",
                )
                selector_status = gr.Markdown("Upload a video to select the watermark region.")
                propainter_dir = gr.Textbox(
                    label="ProPainter directory",
                    placeholder="/path/to/ProPainter",
                )
                output_dir = gr.Textbox(
                    label="Output directory (optional)",
                    placeholder="Defaults to the input video's directory",
                )
                mask_dir = gr.Textbox(
                    label="Per-frame mask directory (optional)",
                    placeholder="Leave empty to use a region mask",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Watermark region")
                custom_region = gr.Checkbox(
                    label="Use custom region",
                    value=False,
                    info="Two preview clicks enable this automatically; manual values still work.",
                )
                with gr.Row():
                    x = gr.Number(label="X", value=0, precision=0)
                    y = gr.Number(label="Y", value=0, precision=0)
                with gr.Row():
                    width = gr.Number(label="Width", value=200, precision=0)
                    height = gr.Number(label="Height", value=100, precision=0)

        with gr.Accordion("Processing settings", open=False):
            with gr.Row():
                temporal_radius = gr.Slider(0, 10, value=2, step=1, label="Temporal radius")
                chunk_size = gr.Slider(1, 96, value=24, step=1, label="Chunk size")
                min_scene_length = gr.Slider(
                    1, 120, value=10, step=1, label="Minimum scene length"
                )
            scene_threshold = gr.Slider(
                -1.0, 1.0, value=0.62, step=0.01, label="Scene-cut threshold"
            )
            motion_compensation = gr.Checkbox(
                label="Optical-flow motion compensation",
                value=True,
            )
            with gr.Row():
                alpha_inpaint_threshold = gr.Slider(
                    0.0, 1.0, value=0.55, step=0.01, label="Alpha inpaint threshold"
                )
                analytic_confidence_min = gr.Slider(
                    0.0, 1.0, value=0.28, step=0.01, label="Minimum analytic confidence"
                )
                residual_dilate = gr.Slider(
                    0, 15, value=3, step=1, label="Residual dilation"
                )
            with gr.Row():
                neighbor_length = gr.Slider(
                    1, 30, value=10, step=1, label="ProPainter neighbor length"
                )
                ref_stride = gr.Slider(
                    1, 30, value=10, step=1, label="ProPainter reference stride"
                )
                resize_ratio = gr.Slider(
                    0.25, 1.0, value=1.0, step=0.05, label="Resize ratio"
                )
            with gr.Row():
                fp16 = gr.Checkbox(label="Use FP16", value=False)
                save_debug = gr.Checkbox(label="Save debug frames", value=False)

        with gr.Accordion("Preflight", open=True):
            preflight_button = gr.Button("Run preflight")
            preflight_status = gr.Markdown(
                "Upload a video and configure ProPainter, then run preflight."
            )

        process_button = gr.Button("Remove watermark", variant="primary")
        status = gr.Markdown("Ready. Processing progress appears above while a run is active.")

        with gr.Row():
            result_video = gr.Video(label="Processed video")
            report_file = gr.File(label="Quality report")

        input_video.change(
            fn=load_preview,
            inputs=[input_video],
            outputs=[preview_image, region_points, selector_status],
        )
        preview_image.select(
            fn=select_region,
            inputs=[preview_image, region_points],
            outputs=[
                region_points,
                x,
                y,
                width,
                height,
                custom_region,
                preview_image,
                selector_status,
            ],
        )
        preflight_button.click(
            fn=run_ui_preflight,
            inputs=[input_video, propainter_dir, output_dir, save_debug],
            outputs=[preflight_status],
        )
        process_button.click(
            fn=process_with_progress,
            inputs=[
                input_video,
                propainter_dir,
                output_dir,
                mask_dir,
                custom_region,
                x,
                y,
                width,
                height,
                temporal_radius,
                chunk_size,
                scene_threshold,
                min_scene_length,
                motion_compensation,
                alpha_inpaint_threshold,
                analytic_confidence_min,
                residual_dilate,
                neighbor_length,
                ref_stride,
                resize_ratio,
                fp16,
                save_debug,
            ],
            outputs=[result_video, report_file, status],
        )

    return app


def main() -> None:
    build_app().launch(show_error=True)


if __name__ == "__main__":
    main()
