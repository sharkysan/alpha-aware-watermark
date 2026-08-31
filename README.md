# Alpha-Aware Temporal Watermark Remover

A maintainable Python pipeline for removing watermarks from **video you own or have permission to edit**.

The project combines alpha-aware inverse compositing, scene-aware temporal analysis, optical-flow alignment, residual AI inpainting with ProPainter, bounded-memory processing, disk-space preflight checks, and an optional local Gradio UI.

> Package metadata is currently `1.1.0`; this README documents the full feature set available on `main`.

## Highlights

- **Alpha-aware recovery** — analytically reconstructs pixels where watermark opacity and confidence allow it.
- **Residual AI inpainting** — sends only uncertain pixels to ProPainter instead of regenerating the whole marked region.
- **Scene-aware processing** — temporal windows stay inside detected visual shots.
- **Optical-flow alignment** — neighbouring frames are warped before temporal background estimation.
- **Bounded RAM usage** — processing is chunked instead of loading the full video into memory.
- **Flexible masks** — use per-frame masks or a rectangular fallback region through the `MaskProvider` strategy.
- **Disk-space preflight** — checks scratch and output capacity before expensive frame extraction/inference begins.
- **Local web UI** — optional Gradio interface for upload, preview, region/mask selection, tuning, processing and result download.
- **OpenCV 4/5 support** — `opencv-python>=4.8,<6`.
- **Production-oriented CI** — Ruff, mypy, pytest/coverage, Bandit, pip-audit, packaging checks and CodeQL.

## How it works

A translucent watermark can be approximated by:

```text
O = αF + (1 - α)B
```

where:

- `O` = observed video pixel
- `α` = watermark opacity
- `F` = watermark colour
- `B` = hidden background

When `α` and `F` can be estimated, the background can be approximated by:

```text
B = (O - αF) / (1 - α)
```

The inverse becomes unstable as `α` approaches `1`, so the pipeline calculates confidence and sends uncertain pixels to ProPainter rather than trusting the analytic reconstruction.

The processing flow is:

```text
Input video
    │
    ▼
Disk-space preflight
    │
    ▼
FFmpeg frame extraction
    │
    ▼
Scene-cut detection
    │
    ▼
Bounded scene chunks
    │
    ├── MaskProvider → support mask
    │
    └── neighbouring frames
              │
              ▼
       optical-flow alignment
              │
              ▼
       temporal background median
              │
              ▼
       alpha + foreground estimate
              │
              ▼
         inverse compositing
              │
       ┌──────┴────────┐
       ▼               ▼
confident pixels   uncertain pixels
analytic recovery  residual AI mask
       │               │
       └──────┬────────┘
              ▼
         ProPainter
              │
              ▼
     original audio restored
              │
              ▼
      cleaned video + CSV report
```

## Installation

Python 3.10 or newer is required.

Create a virtual environment and install the core package:

```bash
pip install -e .
```

For the optional local UI:

```bash
pip install -e ".[ui]"
```

For development tools:

```bash
pip install -e ".[dev]"
```

Install FFmpeg separately and verify it is available:

```bash
ffmpeg -version
```

Install ProPainter separately according to its upstream instructions and licence terms.

## Local Gradio UI

The easiest way to use the project interactively is the optional Gradio UI:

```bash
watermark-remove-ui
```

The UI provides:

- video upload and preview;
- ProPainter and output paths;
- per-frame mask-directory input;
- custom watermark-region coordinates;
- chunk size and temporal radius;
- scene-cut controls;
- optical-flow motion-compensation toggle;
- alpha/confidence thresholds;
- residual-mask dilation;
- ProPainter neighbour/reference settings;
- resize ratio, FP16 and debug-output controls;
- processed-video preview;
- quality-report download.

The UI is intentionally a thin adapter around `PipelineConfig` and `WatermarkRemovalPipeline`, so processing behaviour stays aligned with the CLI rather than being duplicated in a second implementation.

See [`UI.md`](UI.md) for detailed UI setup and security guidance.

## Command-line usage

### With per-frame masks

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --mask-dir ./sam_masks \
  --fp16
```

### With a known region

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --x 1115 --y 540 --w 125 --h 130 \
  --fp16
```

If neither masks nor explicit coordinates are supplied, the pipeline uses its default bottom-right fallback region.

## Processing controls

### Chunk size

```bash
--chunk-size 24
```

Controls how many output frames are processed at once. Smaller values reduce peak RAM; larger values reduce repeated disk reads.

### Temporal radius

```bash
--temporal-radius 2
```

A radius of `2` uses up to two preceding and two following frames from the same detected scene.

### Scene-cut threshold

```bash
--scene-threshold 0.62
```

Hard scene cuts are detected from HSV histogram similarity. Temporal windows never intentionally cross a detected cut.

### Minimum scene length

```bash
--min-scene-length 10
```

Avoids creating extremely short scenes from transient histogram changes.

### Motion compensation

Optical-flow alignment is enabled by default. To disable it:

```bash
--no-motion-compensation
```

### Alpha threshold for AI fallback

```bash
--alpha-inpaint 0.55
```

Higher estimated opacity is treated as less trustworthy for inverse compositing and routed to ProPainter.

### Analytic confidence threshold

```bash
--analytic-confidence-min 0.28
```

Low-confidence pixels are included in the residual inpainting mask.

### Residual dilation

```bash
--residual-dilate 3
```

Adds a small safety border around uncertain pixels to cover anti-aliased edges and glow.

### ProPainter GPU-memory controls

```bash
--neighbor-length 6 \
--ref-stride 14 \
--resize-ratio 0.75 \
--fp16
```

These are independent of the pipeline's CPU/RAM chunk size.

## Disk-space preflight

The pipeline estimates scratch and destination requirements before processing starts.

The scratch estimate accounts for the frame sets materialized during processing, including source, analytic, residual and inpainted frames, with additional allowance when debug images are enabled. The output filesystem also receives a conservative reserve for the encoded/muxed result.

If capacity is insufficient, the run fails early with required and available space reported in human-readable units instead of failing late after frame extraction or inference.

## Mask providers

Mask acquisition is separated from orchestration through a small strategy interface:

```text
MaskProvider
├── RegionMaskProvider
└── DirectoryMaskProvider
```

`DirectoryMaskProvider` can fall back to a region mask when a per-frame mask is missing.

This keeps future integrations isolated from the core pipeline. A direct SAM 2.1 provider could implement the same interface:

```python
class Sam21MaskProvider:
    def get_mask(
        self,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        ...
```

## Memory characteristics

The pipeline does not construct a list containing every decoded video frame.

For each scene chunk it loads approximately:

```text
process frames + left temporal overlap + right temporal overlap
```

For example:

```text
chunk size       = 24 frames
temporal radius  = 2 frames
maximum loaded   ≈ 28 frames
```

Decoded-image memory therefore scales with chunk size and temporal overlap rather than total video duration.

Scratch-disk usage can still be significant because extracted and processed frames are materialized for the external inpainting stage, which is why disk-space preflight is performed before extraction.

## Quality report

Each successful run produces a CSV containing:

- frame index;
- support-mask pixel count;
- residual inpainting pixel count;
- mean estimated alpha;
- mean analytic confidence;
- residual fraction.

This helps identify frames where most of the marked region had to be regenerated rather than analytically recovered.

## Package structure

```text
watermark_remover/
├── __init__.py
├── alpha.py            # alpha estimation and inverse compositing
├── chunks.py           # bounded scene chunk planning
├── cli.py              # CLI argument handling
├── disk.py             # disk-space estimation and preflight checks
├── infra.py            # subprocess + ProPainter adapter
├── mask_providers.py   # MaskProvider strategies
├── masks.py            # mask file/region primitives
├── models.py           # typed configuration and value objects
├── motion.py           # optical-flow alignment
├── pipeline.py         # application orchestration
├── ports.py            # dependency-inversion protocols
├── reporting.py        # CSV quality reporting
├── scenes.py           # hard scene-cut detection
├── ui.py               # optional Gradio UI
└── video.py            # probing, extraction and audio muxing
```

## Tests and quality checks

Run the test suite:

```bash
pytest
```

Coverage:

```bash
pytest --cov=watermark_remover --cov-report=term-missing
```

Static checks:

```bash
ruff check .
mypy watermark_remover
```

The suite covers the deterministic numerical core and orchestration seams, including:

- alpha inverse-compositing math;
- residual-mask behaviour;
- chunk bounds and scene isolation;
- optical-flow alignment;
- mask-provider strategies;
- reporting;
- ProPainter adapter validation;
- pipeline temporal-window clipping;
- disk-space estimation/preflight behaviour;
- UI-to-`PipelineConfig` mapping and pipeline delegation.

GPU/model-heavy tests remain a separate integration concern so normal development and pull requests stay fast and reproducible.

## Continuous integration and security

Every push to `main` and every pull request is checked by GitHub Actions.

CI validates:

- dependency consistency with `pip check`;
- linting with Ruff;
- static typing with mypy;
- tests on Python 3.10 and 3.12;
- branch-aware coverage with a 60% minimum threshold;
- Bandit static security findings;
- known dependency vulnerabilities with `pip-audit`;
- wheel/source-distribution creation;
- package metadata with `twine check`.

A separate CodeQL workflow runs on pushes, pull requests and a weekly schedule. Dependabot monitors Python dependencies and GitHub Actions.

## Design principles

The codebase deliberately keeps responsibilities separated instead of putting the complete workflow into one script:

- **Single Responsibility Principle** — numerical processing, scenes, masks, video I/O, disk checks, external tools, reporting, UI and orchestration live in separate modules.
- **Strategy pattern** — masks are supplied through `MaskProvider` implementations.
- **Dependency inversion** — external commands and inpainting backends are accessed through protocols/adapters.
- **Pure functions where practical** — numerical operations are isolated from filesystem/subprocess effects.
- **Fail fast** — invalid configuration, missing dependencies and insufficient disk space are detected before expensive work.
- **Immutable configuration/value objects** — dataclasses are frozen where appropriate.
- **Bounded resource use** — memory scales with configured chunk size rather than video length.
- **Testable seams** — process runners, pipelines and mask providers can be replaced with test doubles.

## Remaining improvements

Useful next steps include:

- resumable scene/chunk checkpoints;
- structured logging and a per-run model/config manifest;
- automated quality gates and retry policies;
- direct `Sam21MaskProvider` integration;
- an interactive mask editor in the UI;
- a GPU integration-test workflow;
- container/lockfile reproducibility;
- scene-level parallel scheduling where GPU memory permits.

## Limitations

Alpha and watermark colour cannot generally be uniquely recovered from a single composite image. The estimator uses temporal statistics and residual magnitude as a practical approximation.

Optical flow can fail around occlusion, large motion, severe blur, repeated textures and abrupt lighting changes. Scene isolation reduces one major failure mode but cannot eliminate all motion-estimation errors.

No inpainting method can guarantee recovery of information that was never visible in any source frame.

Use this tool only on material you own or have permission to modify.

## Licence

The source code in this repository is licensed under the MIT License; see [`LICENSE`](LICENSE).

Third-party components are governed by their own licences. Installing or invoking ProPainter, SAM 2, SAM 2.1, FFmpeg, Gradio or other external software does **not** make those components MIT-licensed. Review their current licence terms before commercial use, redistribution or deployment.

Security reporting guidance is documented in [`SECURITY.md`](SECURITY.md).
