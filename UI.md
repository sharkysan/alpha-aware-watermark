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

## Available controls

- upload and preview the input video;
- configure the local ProPainter directory;
- choose an output directory;
- use a per-frame mask directory or the pipeline's region mask;
- optionally enter a custom `x`, `y`, `width`, and `height` region;
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
