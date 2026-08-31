# Processing progress

The pipeline emits optional monotonic progress updates without depending on any UI framework.

Current stages are:

- configuration validation;
- disk/prerequisite preflight;
- source-frame extraction;
- scene detection;
- per-frame analytic processing;
- ProPainter residual inpainting;
- audio muxing/output writing;
- quality-report generation;
- completion.

The Gradio UI maps these updates to its native progress indicator. Other callers can pass their own `ProgressReporter` callback to `WatermarkRemovalPipeline` or `process_video`.
