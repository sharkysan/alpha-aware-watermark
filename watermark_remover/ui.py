from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import PipelineConfig, Region
from .pipeline import WatermarkRemovalPipeline
from .presets import preset_values
from .progress import PipelineProgress, ProgressReporter
from .ui_preflight import build_ui_preflight_report

APP_CSS = """
.gradio-container { max-width: 1440px !important; }
.hero-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 14px;
    background: var(--background-fill-secondary);
}
.hero-card h1 { margin: 0 0 6px 0; }
.hero-subtitle { opacity: 0.78; margin-bottom: 10px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    border: 1px solid var(--border-color-primary);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.82rem;
    background: var(--background-fill-primary);
}
.panel-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 14px;
    background: var(--background-fill-secondary);
}
#watermark-selector { min-height: 520px; }
#watermark-selector > div { border-radius: 14px; overflow: hidden; }
.selector-help { font-size: 0.9rem; opacity: 0.78; }
.primary-action button { min-height: 48px; font-weight: 650; }
.compact-status { min-height: 42px; }
"""


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser()


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
    roi_padding: float = 32,
    save_debug: bool = False,
    propainter_python: str | Path | None = None,
) -> PipelineConfig:
    """Translate UI values into the same PipelineConfig used by the CLI."""
    input_path = Path(input_video).expanduser()
    propainter_path = Path(propainter_dir).expanduser()
    destination = _optional_path(output_dir) or input_path.parent
    output_path = destination / f"{input_path.stem}_alpha_clean.mp4"
    report_path = destination / f"{input_path.stem}_alpha_quality.csv"

    region = None
    if custom_region:
        region = Region(x=int(x), y=int(y), width=int(width), height=int(height))

    config = PipelineConfig(
        input_path=input_path,
        propainter_dir=propainter_path,
        output_path=output_path,
        report_path=report_path,
        propainter_python=_optional_path(propainter_python),
        mask_dir=_optional_path(mask_dir),
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
        roi_padding=int(roi_padding),
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
    """Legacy two-click region helper retained for API compatibility and tests."""
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


def region_from_annotation(annotation: dict[str, Any] | None) -> Region | None:
    """Convert the drag-selector's first bounding box into a pipeline Region."""
    if not annotation:
        return None
    boxes = annotation.get("boxes") or []
    if not boxes:
        return None
    box = boxes[0]
    xmin = round(float(box["xmin"]))
    ymin = round(float(box["ymin"]))
    xmax = round(float(box["xmax"]))
    ymax = round(float(box["ymax"]))
    left, right = sorted((xmin, xmax))
    top, bottom = sorted((ymin, ymax))
    return Region(
        x=max(0, left),
        y=max(0, top),
        width=max(1, right - left),
        height=max(1, bottom - top),
    )


def run_ui_preflight(
    input_video: str | Path | None,
    propainter_dir: str | Path | None,
    propainter_python: str | Path | None,
    output_dir: str | Path | None,
    save_debug: bool,
) -> str:
    """Return the rendered preflight report used by the Gradio panel."""
    return build_ui_preflight_report(
        input_video,
        propainter_dir,
        output_dir,
        propainter_python=propainter_python,
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
    roi_padding: float = 32,
    save_debug: bool = False,
    propainter_python: str | Path | None = None,
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
        roi_padding=roi_padding,
        save_debug=save_debug,
        propainter_python=propainter_python,
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
        from gradio_image_annotation import image_annotator  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            'Gradio UI dependencies are not installed. Run: pip install -e ".[ui]"'
        ) from exc

    def load_preview(
        video: str | None,
    ) -> tuple[dict[str, Any] | None, float, float, float, float, bool, str]:
        frame = extract_preview_frame(video)
        if frame is None:
            return None, 0.0, 0.0, 1.0, 1.0, False, "Upload a video to begin."
        return (
            {"image": frame, "boxes": []},
            0.0,
            0.0,
            1.0,
            1.0,
            False,
            "Drag a rectangle around the watermark. You can move or resize it afterwards.",
        )

    def sync_region_from_annotation(
        annotation: dict[str, Any] | None,
    ) -> tuple[float, float, float, float, bool, str]:
        region = region_from_annotation(annotation)
        if region is None:
            return (
                0.0,
                0.0,
                1.0,
                1.0,
                False,
                "Drag a rectangle around the watermark to select the processing region.",
            )
        return (
            float(region.x),
            float(region.y),
            float(region.width),
            float(region.height),
            True,
            (
                f"Selected region: x={region.x}, y={region.y}, "
                f"width={region.width}, height={region.height}."
            ),
        )

    def process_with_progress(
        input_video: str | Path,
        propainter_dir: str | Path,
        propainter_python: str | Path | None,
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
        roi_padding: float,
        save_debug: bool,
        progress: Any = gr.Progress(),
    ) -> tuple[str, str, str]:
        def report(update: PipelineProgress) -> None:
            progress(update.fraction, desc=update.message)

        return process_video(
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
            roi_padding=roi_padding,
            save_debug=save_debug,
            propainter_python=propainter_python,
            progress_reporter=report,
        )

    with gr.Blocks(
        title="Alpha-Aware Watermark",
        css=APP_CSS,
        theme=gr.themes.Soft(),
    ) as app:
        gr.HTML(
            """
            <div class="hero-card">
              <h1>Alpha-Aware Watermark</h1>
              <div class="hero-subtitle">
                Scene-aware temporal watermark removal with analytic recovery and residual AI inpainting.
              </div>
              <div class="chips">
                <span class="chip">Alpha-aware</span>
                <span class="chip">Scene-aware</span>
                <span class="chip">Optical flow</span>
                <span class="chip">ProPainter</span>
                <span class="chip">ROI accelerated</span>
                <span class="chip">Local processing</span>
              </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Process"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, elem_classes=["panel-card"]):
                        gr.Markdown("### 1. Video and watermark region")
                        input_video = gr.Video(label="Input video", sources=["upload"])
                        gr.Markdown(
                            "**Drag directly on the preview to draw the watermark rectangle.** "
                            "Drag the box to move it or use its handles to resize it.",
                            elem_classes=["selector-help"],
                        )
                        region_selector = image_annotator(
                            value=None,
                            image_type="numpy",
                            single_box=True,
                            disable_edit_boxes=True,
                            show_download_button=False,
                            show_clear_button=False,
                            show_remove_button=True,
                            boxes_alpha=0.18,
                            box_thickness=2,
                            box_selected_thickness=3,
                            box_min_size=4,
                            height=520,
                            label="Watermark region",
                            elem_id="watermark-selector",
                        )
                        selector_status = gr.Markdown(
                            "Upload a video to select the watermark region.",
                            elem_classes=["compact-status"],
                        )

                    with gr.Column(scale=4, elem_classes=["panel-card"]):
                        gr.Markdown("### 2. Region and runtime")
                        custom_region = gr.Checkbox(
                            label="Use custom region",
                            value=False,
                            info="Enabled automatically when a rectangle is drawn.",
                        )
                        with gr.Row():
                            x = gr.Number(label="X", value=0, precision=0)
                            y = gr.Number(label="Y", value=0, precision=0)
                        with gr.Row():
                            width = gr.Number(label="Width", value=1, precision=0)
                            height = gr.Number(label="Height", value=1, precision=0)

                        propainter_dir = gr.Textbox(
                            label="ProPainter directory",
                            placeholder=r"C:\Work\ProPainter",
                        )
                        propainter_python = gr.Textbox(
                            label="ProPainter Python executable",
                            placeholder=(
                                r"C:\Users\you\miniconda3\envs\propainter\python.exe "
                                "(blank = UI Python)"
                            ),
                            info=(
                                "Use the Python from the Conda/venv where ProPainter dependencies "
                                "are installed."
                            ),
                        )
                        output_dir = gr.Textbox(
                            label="Output directory (optional)",
                            placeholder="Defaults to the input video's directory",
                        )
                        mask_dir = gr.Textbox(
                            label="Per-frame mask directory (optional)",
                            placeholder="Leave empty to use the selected rectangle",
                        )

                        gr.Markdown("### 3. Preflight")
                        preflight_button = gr.Button("Run preflight")
                        preflight_status = gr.Markdown(
                            "Preflight checks video, FFmpeg, disk space, ProPainter, and its Python environment.",
                            elem_classes=["compact-status"],
                        )
                        process_button = gr.Button(
                            "Remove watermark",
                            variant="primary",
                            elem_classes=["primary-action"],
                        )
                        status = gr.Markdown(
                            "Ready. Processing progress appears while a run is active.",
                            elem_classes=["compact-status"],
                        )

                with gr.Accordion("Processing settings", open=False):
                    processing_preset = gr.Radio(
                        choices=["Fast", "Balanced", "High Quality"],
                        value="Balanced",
                        label="Processing preset",
                        info="Fast prioritizes throughput; High Quality preserves maximum detail.",
                    )
                    with gr.Row():
                        temporal_radius = gr.Slider(
                            0, 10, value=2, step=1, label="Temporal radius"
                        )
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
                            0.0,
                            1.0,
                            value=0.55,
                            step=0.01,
                            label="Alpha inpaint threshold",
                        )
                        analytic_confidence_min = gr.Slider(
                            0.0,
                            1.0,
                            value=0.28,
                            step=0.01,
                            label="Minimum analytic confidence",
                        )
                        residual_dilate = gr.Slider(
                            0, 15, value=3, step=1, label="Residual dilation"
                        )
                    with gr.Row():
                        neighbor_length = gr.Slider(
                            1, 30, value=8, step=1, label="ProPainter neighbor length"
                        )
                        ref_stride = gr.Slider(
                            1, 30, value=10, step=1, label="ProPainter reference stride"
                        )
                        resize_ratio = gr.Slider(
                            0.25, 1.0, value=0.75, step=0.05, label="Resize ratio"
                        )
                    with gr.Row():
                        fp16 = gr.Checkbox(label="Use FP16", value=True)
                        roi_padding = gr.Slider(
                            0,
                            128,
                            value=32,
                            step=4,
                            label="ROI padding",
                            info="Extra pixels around residual masks sent to ProPainter.",
                        )
                        save_debug = gr.Checkbox(label="Save debug frames", value=False)

                gr.Markdown("### Result")
                with gr.Row():
                    result_video = gr.Video(label="Processed video")
                    report_file = gr.File(label="Quality report")

            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ### Workflow
                    1. Upload a video.
                    2. Drag a rectangle around the watermark in the preview.
                    3. Select the ProPainter directory and, when using a separate Conda/venv,
                       its Python executable.
                    4. Pick Fast, Balanced, or High Quality and adjust individual settings if needed.
                    5. Run preflight, then start processing.

                    ProPainter receives only the padded union of residual inpainting masks instead of
                    full-resolution frames. The ROI result is composited back into the analytic frames
                    before the original audio is restored.
                    """
                )

        input_video.change(
            fn=load_preview,
            inputs=[input_video],
            outputs=[
                region_selector,
                x,
                y,
                width,
                height,
                custom_region,
                selector_status,
            ],
        )
        region_selector.change(
            fn=sync_region_from_annotation,
            inputs=[region_selector],
            outputs=[x, y, width, height, custom_region, selector_status],
        )
        processing_preset.change(
            fn=preset_values,
            inputs=[processing_preset],
            outputs=[
                temporal_radius,
                chunk_size,
                motion_compensation,
                neighbor_length,
                ref_stride,
                resize_ratio,
                fp16,
                roi_padding,
            ],
        )
        preflight_button.click(
            fn=run_ui_preflight,
            inputs=[
                input_video,
                propainter_dir,
                propainter_python,
                output_dir,
                save_debug,
            ],
            outputs=[preflight_status],
        )
        process_button.click(
            fn=process_with_progress,
            inputs=[
                input_video,
                propainter_dir,
                propainter_python,
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
                roi_padding,
                save_debug,
            ],
            outputs=[result_video, report_file, status],
        )

    return app


def main() -> None:
    build_app().launch(show_error=True)


if __name__ == "__main__":
    main()
