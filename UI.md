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

## Drag-to-select watermark region

After uploading a video, the UI extracts the first frame and opens it in a bounding-box selector.

1. Drag a rectangle around the watermark.
2. Move the selected rectangle by dragging it, or resize it with its handles.
3. The UI synchronizes the rectangle into `x`, `y`, `width`, and `height` and enables the custom region automatically.

The selector keeps at most one active box. Remove it and draw another box to replace the selection. The numeric region fields remain editable for precise manual adjustment.

If a per-frame mask directory is supplied, those masks still take precedence and the region remains the fallback behavior used by the pipeline.

## ProPainter Python environment

The UI can run ProPainter with a Python executable different from the Python that runs Gradio. This is recommended when ProPainter is installed in a dedicated Conda environment or virtual environment.

On Windows, for example:

```text
ProPainter directory:
C:\Work\ProPainter

ProPainter Python executable:
C:\Users\you\miniconda3\envs\propainter\python.exe
```

Leave **ProPainter Python executable** blank to use the same Python interpreter as the UI.

The preflight runs the selected interpreter from the ProPainter directory and verifies that core runtime imports such as `imageio`, `torch`, `torchvision`, `cv2`, and `numpy` succeed. It also reports the detected Python and PyTorch versions. A missing dependency therefore fails before the expensive video-processing phase starts.

## Available controls

- upload and preview the input video;
- draw, move, and resize a rectangular watermark region directly on a preview frame;
- configure the local ProPainter directory;
- select the Python executable for the ProPainter environment;
- choose an output directory;
- use a per-frame mask directory or the pipeline's region mask;
- manually edit custom `x`, `y`, `width`, and `height` values;
- tune temporal radius and chunk size;
- tune scene-cut detection;
- enable or disable optical-flow motion compensation;
- tune alpha confidence and residual-mask thresholds;
- tune ProPainter neighbour length, reference stride, and resize ratio;
- enable FP16 and debug-frame output;
- run preflight checks for FFmpeg, disk capacity, ProPainter, and its Python environment;
- preview the processed video;
- download the quality CSV report.

The UI does not duplicate processing logic. It translates form values into `PipelineConfig` and delegates the run to `WatermarkRemovalPipeline`, so CLI and UI behaviour stay aligned.

## Security and deployment

The UI is intended for local use. Do not expose it directly to the public internet without adding appropriate authentication, network controls, upload limits, and deployment hardening.
