# Local Gradio UI

The repository includes an optional local web UI that wraps the same `PipelineConfig` and `WatermarkRemovalPipeline` used by the CLI.

## Install

```bash
pip install -e ".[ui]"
```

The UI extra installs Gradio and the Gradio 6 compatible bounding-box selector used for drag-to-select watermark regions. FFmpeg and ProPainter are still external prerequisites.

## Launch

```bash
watermark-remove-ui
```

Gradio starts a local web application and prints its local address in the terminal.

## Interactive watermark region selection

After uploading a video, the UI extracts the first frame and loads it into the watermark-region selector.

1. Drag directly over the preview to draw a rectangle around the watermark.
2. Drag the rectangle to move it, or use its handles to resize it.
3. The UI automatically fills `x`, `y`, `width`, and `height` and enables the custom region.
4. Remove the box to clear the selection, or edit the numeric fields for exact manual adjustment.

The selector is intentionally limited to one box because the current pipeline accepts one rectangular fallback region. If a per-frame mask directory is supplied, those masks still take precedence and the rectangle remains the fallback behavior used by the pipeline.

## UI layout

The main **Process** tab follows the visual workflow shown in the project overview screenshot:

- video upload and a large drag-to-select preview on the left;
- region coordinates, ProPainter/output paths, preflight and the primary processing action on the right;
- advanced temporal, scene, optical-flow, alpha and ProPainter controls in a collapsible section;
- processed-video preview and quality-report download below the processing controls.

An **About** tab gives a short workflow summary without duplicating the detailed processing documentation.

## Available controls

- upload and preview the input video;
- drag, move and resize a rectangular watermark region directly on a preview frame;
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
