# Local Gradio UI

The repository includes an optional local web UI that wraps the same `PipelineConfig` and `WatermarkRemovalPipeline` used by the CLI.

## Install

```bash
pip install -e ".[ui]"
```

FFmpeg and ProPainter are still external prerequisites.

## Launch

```bash
watermark-remove-ui
```

Gradio starts a local web application and prints its local address in the terminal.

## Interactive watermark region selection

After uploading a video, the UI extracts the first frame and shows it in the watermark-region selector.

1. Click one corner of the watermark.
2. Click the opposite corner.
3. The UI fills `x`, `y`, `width`, and `height`, enables the custom region, and draws the selected rectangle on the preview.

A third click starts a new selection. The numeric region fields remain editable for precise manual adjustment.

If a per-frame mask directory is supplied, those masks still take precedence and the region remains the fallback behavior used by the pipeline.

## Available controls

- upload and preview the input video;
- select a rectangular watermark region directly on a preview frame;
- configure the local ProPainter directory;
- choose an output directory;
- use a per-frame mask directory or the pipeline's region mask;
- manually edit custom `x`, `y`, `width`, and `height` values;
- tune temporal radius and chunk size;
- tune scene-cut detection;
- enable or disable optical-flow motion compensation;
- tune alpha confidence and residual-mask thresholds;
- tune ProPainter neighbour length, reference stride, and resize ratio;
- enable FP16 and debug-frame output;
- preview the processed video;
- download the quality CSV report.

The UI does not duplicate processing logic. It translates form values into `PipelineConfig` and delegates the run to `WatermarkRemovalPipeline`, so CLI and UI behaviour stay aligned.

## Security and deployment

The UI is intended for local use. Do not expose it directly to the public internet without adding appropriate authentication, network controls, upload limits, and deployment hardening.
