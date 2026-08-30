# Alpha-Aware Temporal Watermark Remover

A maintainable reference implementation for removing a watermark from **video you own or have permission to edit**.

The pipeline combines alpha-aware inverse compositing with temporal video inpainting. Version 1.1 adds bounded-memory processing, formal mask-provider strategies, scene isolation, and optical-flow-aligned temporal background estimation.

## What changed in 1.1

### Bounded chunked processing

The original implementation extracted frames to disk and then loaded the entire video into RAM. The new pipeline processes one scene in bounded chunks with only the temporal overlap required by the configured radius.

For example:

```text
chunk size       = 24 frames
temporal radius  = 2 frames
maximum loaded   ≈ 28 frames
```

Memory usage is therefore approximately proportional to:

```text
chunk_size + 2 × temporal_radius
```

rather than total video length.

The extracted frame files still remain on scratch disk because ProPainter consumes frame directories, but decoded image memory is bounded.

### `MaskProvider` strategy

Mask acquisition is now separated from the pipeline through a small strategy interface:

```text
MaskProvider
├── RegionMaskProvider
└── DirectoryMaskProvider
```

`DirectoryMaskProvider` can use a `RegionMaskProvider` as a fallback when a per-frame mask is missing.

This makes it straightforward to add future providers such as:

```text
Sam21MaskProvider
InteractiveMaskProvider
RemoteSegmentationMaskProvider
```

without changing the core processing workflow.

### Scene-aware temporal processing

Hard scene cuts are detected before alpha/background estimation. Temporal windows are clipped to the current visual shot, preventing unrelated frames from different scenes from contaminating the temporal background estimate.

### Optical-flow-aligned temporal median

Before computing a temporal median, neighbouring frames are warped into the current frame's coordinates using dense Farneback optical flow.

Without alignment:

```text
neighboring frames
      ↓
raw temporal median
      ↓
motion ghosting / smeared background estimate
```

With alignment:

```text
neighboring frames
      ↓
optical-flow warp to current frame
      ↓
aligned temporal median
      ↓
cleaner background estimate
```

This improves alpha estimation when the camera or scene is moving.

## Design principles

The codebase deliberately applies a small set of maintainability patterns rather than putting the entire workflow in one script.

- **Single Responsibility Principle** — alpha estimation, scene detection, motion alignment, chunking, masks, video I/O, external-process integration, reporting, and orchestration are separate modules.
- **Strategy pattern** — watermark masks are supplied through `MaskProvider` implementations.
- **Dependency inversion** — external commands and inpainting backends are accessed through protocols/adapters.
- **Pure functions where practical** — numerical processing is isolated from filesystem and subprocess operations.
- **Fail fast** — invalid configuration and missing dependencies are checked before expensive inference.
- **Immutable configuration/value objects** — dataclasses are frozen where appropriate.
- **Bounded resource use** — chunk loading is explicitly constrained instead of scaling with video length.
- **Scene isolation** — temporal algorithms never intentionally cross detected shot boundaries.
- **Testable seams** — process runners and mask providers can be replaced with test doubles.

## Package structure

```text
watermark_remover/
├── __init__.py
├── alpha.py            # alpha estimation and inverse compositing
├── chunks.py           # bounded scene chunk planning
├── cli.py              # CLI argument handling
├── infra.py            # subprocess + ProPainter adapter
├── mask_providers.py   # MaskProvider strategies
├── masks.py            # mask file/region primitives
├── models.py           # typed configuration and value objects
├── motion.py           # optical-flow alignment
├── pipeline.py         # application orchestration
├── ports.py            # dependency-inversion protocols
├── reporting.py        # CSV quality reporting
├── scenes.py           # hard scene-cut detection
└── video.py            # probing, extraction, audio muxing

tests/
├── test_alpha.py
├── test_chunks.py
├── test_infra.py
├── test_mask_providers.py
├── test_masks.py
├── test_models.py
├── test_motion.py
├── test_pipeline.py
├── test_reporting.py
└── test_scenes.py
```

## Processing technique

A translucent watermark can be approximated by the compositing equation:

```text
O = αF + (1 - α)B
```

where:

- `O` = observed video pixel
- `α` = watermark opacity
- `F` = watermark colour
- `B` = hidden background

When `α` and `F` can be estimated, the original background can be approximated by:

```text
B = (O - αF) / (1 - α)
```

The inverse becomes numerically unstable as `α` approaches `1`, so the pipeline calculates a confidence map and sends uncertain pixels to ProPainter rather than trusting the analytic result.

The current processing architecture is:

```text
Input video
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
    ├── MaskProvider → precise support mask
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
      cleaned output + CSV
```

For the strongest mask quality, the intended higher-level architecture remains:

```text
SAM 2.1 exact mask propagation
        ↓
MaskProvider
        ↓
scene-aware optical-flow background estimation
        ↓
alpha-aware analytic recovery
        ↓
small residual uncertainty mask
        ↓
ProPainter temporal inpainting
```

## Installation

Create a virtual environment and install the package:

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

Install FFmpeg separately and verify:

```bash
ffmpeg -version
```

Install ProPainter separately according to its upstream project instructions.

## Running

### With per-frame SAM masks

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --mask-dir ./sam_masks \
  --fp16
```

### With a known fallback region

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --x 1115 --y 540 --w 125 --h 130 \
  --fp16
```

If neither masks nor coordinates are provided, the fallback region remains tuned to the original bottom-right example.

## New performance and quality controls

### Chunk size

```bash
--chunk-size 24
```

Controls how many output frames are processed at once. Smaller values reduce peak RAM; larger values reduce repeated disk reads.

The temporal overlap is added automatically and is clipped at scene boundaries.

### Temporal radius

```bash
--temporal-radius 2
```

A radius of `2` uses up to two preceding and two following frames from the same scene.

### Scene-cut threshold

```bash
--scene-threshold 0.62
```

The detector compares HSV histograms. A hard cut starts a new temporal context.

### Minimum scene length

```bash
--min-scene-length 10
```

Avoids creating extremely short scenes from transient histogram changes.

### Disable motion compensation

Optical-flow alignment is enabled by default. To compare behaviour or reduce CPU work:

```bash
--no-motion-compensation
```

## Alpha/inpainting controls

### Alpha threshold for AI fallback

```bash
--alpha-inpaint 0.55
```

Higher estimated opacity becomes less trustworthy for inverse compositing and is sent to ProPainter.

### Analytic confidence threshold

```bash
--analytic-confidence-min 0.28
```

Low-confidence pixels are included in the residual inpainting mask.

### Residual dilation

```bash
--residual-dilate 3
```

Adds a small safety border for anti-aliased edges and glow.

## GPU-memory controls for ProPainter

```bash
--neighbor-length 6 \
--ref-stride 14 \
--resize-ratio 0.75 \
--fp16
```

These are independent of the pipeline's CPU/RAM chunk size.

## Tests

Run the unit/regression suite:

```bash
pytest
```

The current suite contains **32 tests** and covers:

- alpha inverse-compositing math
- residual-mask threshold behaviour
- analytic blending
- typed configuration validation
- chunk bounds and temporal overlap
- scene boundary isolation
- hard scene-cut detection
- optical-flow alignment reducing translation error
- mask-provider strategy/fallback behaviour
- region and directory mask primitives
- quality CSV generation
- ProPainter adapter validation
- pipeline temporal-window clipping

Coverage:

```bash
pytest --cov=watermark_remover --cov-report=term-missing
```

Static checks, when development dependencies are installed:

```bash
ruff check .
mypy watermark_remover
```

## Testing philosophy

Fast tests do not download large models or require a GPU. The deterministic numerical core and orchestration boundaries are tested locally with synthetic images and fakes.

GPU/model-heavy tests should be a separate integration layer so normal development and pull requests remain fast and reproducible.

## Memory characteristics

The pipeline no longer constructs a Python list containing every decoded video frame.

For each scene chunk it loads only:

```text
process frames + left temporal overlap + right temporal overlap
```

For a 24-frame chunk and radius 2, a normal interior chunk loads at most 28 decoded images.

This makes long-video processing much more predictable. Scratch-disk requirements can still be significant because extracted and processed frames are materialized for the external inpainting stage.

## Scene-aware behaviour

Temporal background estimation is intentionally constrained to the detected scene:

```text
Scene A frames | CUT | Scene B frames
      ↑                 ↑
  windows stay       windows stay
  inside Scene A     inside Scene B
```

No temporal median should include frames from both sides of a hard cut.

## `MaskProvider` extension example

A future SAM provider only needs to implement:

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

The alpha and inpainting pipeline does not need to know how that mask was produced.

## Quality report

Each run produces a CSV containing:

- frame index
- support-mask pixel count
- residual inpainting pixel count
- mean estimated alpha
- mean analytic confidence
- residual fraction

This helps identify frames where most of the marked region had to be regenerated rather than analytically recovered.

## Remaining production-grade improvements

The next improvements worth considering are:

- structured logging and a per-run model/config manifest;
- disk-space preflight checks;
- resumable scene/chunk checkpoints;
- automated quality gates and retry policies;
- direct `Sam21MaskProvider` integration;
- a GPU integration-test workflow;
- container/lockfile reproducibility;
- scene-level parallel scheduling where GPU memory permits.

## Important limitations

Alpha and watermark colour cannot generally be uniquely recovered from one composite image. The current estimator uses temporal statistics and residual magnitude as a practical approximation.

Optical flow also has failure modes around occlusion, large motion, severe blur, repeated textures, and abrupt lighting changes. Scene isolation reduces one major failure mode but does not eliminate all motion-estimation errors.

No inpainting method can guarantee recovery of information that was never visible in any source frame.

Use this tool only on material you own or have permission to modify.

## Continuous integration and security

Every push to `main` and every pull request is checked by GitHub Actions.

The CI workflow validates:

- package installation and dependency consistency with `pip check`;
- formatting/lint rules with Ruff;
- static typing with mypy;
- the test suite on Python 3.10 and 3.12;
- branch-aware test coverage with a 60% minimum threshold;
- Python security findings with Bandit;
- known dependency vulnerabilities with `pip-audit`;
- wheel and source-distribution creation with `python -m build`;
- package metadata with `twine check`.

A separate CodeQL workflow runs on pushes, pull requests, and weekly to provide
an additional source-code security analysis layer. Dependabot checks both
Python dependencies and GitHub Actions weekly.

The package build produced by CI is uploaded as a workflow artifact, but it is
not automatically published to PyPI.

## Licence

The source code in this repository is licensed under the MIT License; see
[`LICENSE`](LICENSE).

Third-party components are governed by their own licences. In particular,
installing or invoking ProPainter, SAM 2, SAM 2.1, FFmpeg, or other external
software does **not** make those components MIT-licensed. Review their current
licence terms before commercial use, redistribution, or deployment.

Security reporting guidance is documented in [`SECURITY.md`](SECURITY.md).
