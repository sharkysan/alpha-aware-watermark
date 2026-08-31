from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import PipelineConfig, Region
from .pipeline import WatermarkRemovalPipeline


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
        region = Region(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
        )

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
    result = WatermarkRemovalPipeline(config).run()
    return str(result), str(config.report_path), "Processing completed successfully."


def build_app() -> Any:
    """Build the optional local Gradio application."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            'Gradio UI dependencies are not installed. Run: pip install -e ".[ui]"'
        ) from exc

    with gr.Blocks(title="Alpha-Aware Watermark Remover") as app:
        gr.Markdown(
            "# Alpha-Aware Watermark Remover\n"
            "Local UI for the existing scene-aware, temporal processing pipeline."
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="Input video", sources=["upload"])
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
                    info="Otherwise the pipeline's default bottom-right region is used.",
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
                -1.0,
                1.0,
                value=0.62,
                step=0.01,
                label="Scene-cut threshold",
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

        process_button = gr.Button("Remove watermark", variant="primary")
        status = gr.Markdown("Ready.")

        with gr.Row():
            result_video = gr.Video(label="Processed video")
            report_file = gr.File(label="Quality report")

        process_button.click(
            fn=process_video,
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
