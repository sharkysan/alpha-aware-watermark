# Alpha-Aware Temporal Watermark Remover

Version **1.3.0** — a maintainable Python pipeline for removing watermarks from **video you own or have permission to edit**.

The project combines alpha-aware inverse compositing, scene-aware temporal analysis, optical-flow alignment, residual AI inpainting with ProPainter, bounded-memory processing, disk-space preflight checks, resumable checkpoints, and an optional local Gradio UI.

## What’s new in 1.3.0

- **Resumable chunk checkpoints** with `--resume` and optional `--checkpoint-dir`.
- **Interactive watermark-region selection** in the Gradio UI using two preview clicks.
- **UI preflight readiness panel** for input, ProPainter and disk checks before processing.
- **Structured progress reporting** surfaced in the Gradio UI.
- **Filesystem-aware disk preflight** when scratch and output paths share capacity.
- **Expanded checkpoint, UI, progress and disk-preflight tests**.
- **Gradio UI overview screenshot** in the README.

See [`RELEASE_NOTES_1.3.0.md`](RELEASE_NOTES_1.3.0.md) for the full release notes.

## Highlights

- **Alpha-aware recovery** — analytically reconstructs pixels where watermark opacity and confidence allow it.
- **Residual AI inpainting** — sends only uncertain pixels to ProPainter instead of regenerating the whole marked region.
- **Scene-aware processing** — temporal windows stay inside detected visual shots.
- **Optical-flow alignment** — neighbouring frames are warped before temporal background estimation.
- **Bounded RAM usage** — processing is chunked instead of loading the full video into memory.
- **Resumable processing** — completed analytic/residual chunks can be reused after interruptions.
- **Flexible masks** — per-frame masks or a rectangular fallback region through the `MaskProvider` strategy.
- **Disk-space preflight** — checks scratch and output capacity before expensive work begins.
- **Local web UI** — optional Gradio interface built as a thin adapter over the same pipeline used by the CLI.
- **Production-oriented CI** — Ruff, mypy, pytest/coverage, Bandit, pip-audit, packaging checks and CodeQL.

## How it works

A translucent watermark can be approximated by:

```text
O = αF + (1 - α)B
```

where `O` is the observed pixel, `α` the watermark opacity, `F` the watermark colour and `B` the hidden background.

When `α` and `F` can be estimated:

```text
B = (O - αF) / (1 - α)
```

The inverse becomes unstable as `α` approaches `1`, so the pipeline calculates confidence and routes uncertain pixels to ProPainter instead of trusting the analytic reconstruction.

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

Core package:

```bash
pip install -e .
```

Optional UI:

```bash
pip install -e ".[ui]"
```

Development tools:

```bash
pip install -e ".[dev]"
```

Install FFmpeg separately and verify it is available:

```bash
ffmpeg -version
```

Install ProPainter separately according to its upstream instructions and licence terms.

## Local Gradio UI

Launch the optional UI with:

```bash
watermark-remove-ui
```

The UI exposes video upload/preview, interactive two-click watermark-region selection, ProPainter and output paths, per-frame masks, scene/chunk/temporal controls, motion compensation, alpha/confidence tuning, preflight readiness checks, live processing progress, ProPainter memory controls, FP16/debug options, processed-video preview and quality-report download.

<p align="center">
  <img src="docs/images/gradio-ui-overview.png" alt="Alpha-Aware Watermark Remover Gradio UI overview" width="100%">
</p>

The UI deliberately delegates to `PipelineConfig` and `WatermarkRemovalPipeline`; processing logic is not duplicated.

See [`UI.md`](UI.md) for detailed setup and deployment/security guidance.

## Command-line usage

With per-frame masks:

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --mask-dir ./sam_masks \
  --fp16
```

With a known region:

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --x 1115 --y 540 --w 125 --h 130 \
  --fp16
```

Resume a long-running job after interruption:

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --resume
```

Use a custom checkpoint parent directory:

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --resume \
  --checkpoint-dir /fast/local/checkpoints
```

See [`CHECKPOINTS.md`](CHECKPOINTS.md) for checkpoint lifecycle, invalidation and safety details.

If neither masks nor explicit coordinates are supplied, the pipeline uses its default bottom-right fallback region.

## Key processing controls

```bash
--chunk-size 24
--temporal-radius 2
--scene-threshold 0.62
--min-scene-length 10
--alpha-inpaint 0.55
--analytic-confidence-min 0.28
--residual-dilate 3
--neighbor-length 6
--ref-stride 14
--resize-ratio 0.75
--fp16
--resume
```

Optical-flow alignment is enabled by default; disable it with:

```bash
--no-motion-compensation
```

## Disk-space preflight

Before processing, the pipeline estimates scratch and destination requirements. The estimate accounts for materialized source/intermediate frames, optional debug output and a conservative output reserve.

Shared filesystems are accounted for as shared capacity, preventing scratch and output requirements from being double-counted independently when they consume the same disk.

If capacity is insufficient, the run fails early with required and available space reported in human-readable units instead of failing late after extraction or inference.

## Resumable checkpoints

With `--resume`, completed analytic/residual chunks are persisted beneath the output directory by default and reused when the same input, masks/region and relevant processing settings are detected again.

Checkpoint state is fingerprinted and incomplete outputs are rejected. After a fully successful run, the per-input checkpoint data is removed automatically.

The current implementation checkpoints the analytic/residual recovery phase. Source-frame extraction and the final ProPainter pass are repeated after restart.

## Mask providers

Mask acquisition is separated from orchestration:

```text
MaskProvider
├── RegionMaskProvider
└── DirectoryMaskProvider
```

`DirectoryMaskProvider` can fall back to a region mask when a per-frame mask is missing. This keeps future integrations such as a direct SAM 2.1 provider isolated from the core pipeline.

## Memory characteristics

For each scene chunk the pipeline loads approximately:

```text
process frames + left temporal overlap + right temporal overlap
```

For example, a chunk size of 24 with temporal radius 2 loads roughly 28 decoded frames for a normal interior chunk. RAM use therefore scales with chunk size rather than total video duration.

## Quality report

Each successful run produces a CSV containing frame index, support-mask pixel count, residual inpainting pixel count, mean estimated alpha, mean analytic confidence and residual fraction.

## Package structure

```text
watermark_remover/
├── alpha.py            # alpha estimation and inverse compositing
├── checkpoints.py      # resumable analytic/residual chunk checkpoints
├── chunks.py           # bounded scene chunk planning
├── cli.py              # CLI argument handling
├── disk.py             # disk-space estimation and preflight
├── infra.py            # subprocess + ProPainter adapter
├── mask_providers.py   # MaskProvider strategies
├── masks.py            # mask primitives
├── models.py           # typed configuration/value objects
├── motion.py           # optical-flow alignment
├── pipeline.py         # orchestration
├── ports.py            # protocols
├── progress.py         # structured progress events/reporters
├── reporting.py        # CSV quality reporting
├── scenes.py           # hard scene-cut detection
├── ui.py               # optional Gradio UI
├── ui_preflight.py     # UI readiness/preflight reporting
└── video.py            # probing, extraction and audio muxing
```

## Tests and quality checks

```bash
pytest
pytest --cov=watermark_remover --cov-report=term-missing
ruff check .
mypy watermark_remover
```

The suite covers alpha math, residual masks, chunk/scene isolation, optical flow, mask providers, reporting, ProPainter adapter validation, pipeline temporal windows, disk preflight, resumable checkpoints, UI preflight, interactive region selection and progress delegation.

## Continuous integration and security

Every push to `main` and pull request is checked with dependency consistency, Ruff, mypy, tests on Python 3.10 and 3.12, branch coverage, Bandit, pip-audit, package build/metadata checks and CodeQL. Dependabot monitors Python dependencies and GitHub Actions.

## Remaining improvements

Useful next steps include checkpointing frame extraction and the final ProPainter phase, direct `Sam21MaskProvider` integration, structured run manifests/logging, GPU integration tests, and container/lockfile reproducibility.

## Limitations

Alpha and watermark colour cannot generally be uniquely recovered from a single composite image. Optical flow can fail around occlusion, large motion, severe blur, repeated textures and abrupt lighting changes. No inpainting method can guarantee recovery of information that was never visible in any source frame.

Use this tool only on material you own or have permission to modify.

## Licence

The source code in this repository is licensed under the MIT License; see [`LICENSE`](LICENSE).

Third-party components are governed by their own licences. Installing or invoking ProPainter, SAM 2/2.1, FFmpeg, Gradio or other external software does **not** make those components MIT-licensed. Review their current licence terms before commercial use, redistribution or deployment.

Security reporting guidance is documented in [`SECURITY.md`](SECURITY.md).
