# Alpha-Aware Watermark Remover 1.2.0

Version 1.2.0 focuses on usability, safer long-running processing, and broader runtime compatibility while preserving the scene-aware, alpha-aware temporal pipeline introduced in 1.1.

## Highlights

### Optional Gradio UI

A new local web UI makes the pipeline accessible without assembling a long CLI command.

Install and launch it with:

```bash
pip install -e ".[ui]"
watermark-remove-ui
```

The UI supports:

- video upload and preview;
- ProPainter, output and mask paths;
- rectangular watermark-region selection;
- chunk, temporal and scene controls;
- optical-flow motion compensation;
- alpha/confidence thresholds and residual dilation;
- ProPainter neighbour/reference tuning;
- resize ratio, FP16 and debug output;
- processed-video preview and quality-report download.

The UI is intentionally a thin adapter over `PipelineConfig` and `WatermarkRemovalPipeline`, keeping CLI and UI processing behaviour aligned.

### Disk-space preflight checks

The pipeline now estimates scratch and output storage before expensive processing begins.

Preflight checks account for source/intermediate frame materialization, debug-frame overhead and output reserve. When capacity is insufficient, processing fails early with required and available space reported in human-readable units instead of failing after extraction or inference has already consumed time.

### OpenCV 5 support

The supported OpenCV range is now:

```text
opencv-python>=4.8,<6
```

This retains OpenCV 4 compatibility while allowing OpenCV 5.x installations.

## Quality and maintenance

- Added focused tests for disk-space estimation and failure reporting.
- Added tests for UI-to-`PipelineConfig` mapping and pipeline delegation.
- Kept Gradio as an optional dependency so CLI/development installs remain lightweight.
- CI continues to run Ruff, mypy, pytest with branch coverage, Bandit, pip-audit, package builds and metadata validation on supported Python versions.
- GitHub Actions dependencies were updated to current major versions.

## Installation

Core CLI:

```bash
pip install -e .
```

Optional UI:

```bash
pip install -e ".[ui]"
```

Development environment:

```bash
pip install -e ".[dev]"
```

FFmpeg and ProPainter remain external dependencies and must be installed separately according to their upstream instructions and licence terms.

## Upgrade notes

There are no intentional breaking changes to the existing CLI workflow in 1.2.0. Existing region-mask and per-frame-mask usage should continue to work as before.

Users who pin OpenCV may keep an OpenCV 4.x version; upgrading to 5.x is optional.

## Known limitations

- ProPainter is still an external project with its own installation requirements and licence.
- Direct SAM 2.1 mask propagation is not yet integrated into the package.
- The Gradio UI is designed primarily for trusted local use; review `UI.md` before exposing it on a network.
- Scratch-disk use can remain substantial for long or high-resolution videos even though RAM use is bounded.

## Next candidates

Potential follow-up work includes resumable scene/chunk checkpoints, an interactive mask editor, direct `Sam21MaskProvider` integration, structured run manifests/logging, and GPU integration testing.
