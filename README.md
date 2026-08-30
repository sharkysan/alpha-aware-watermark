# Alpha-Aware Temporal Watermark Remover

A maintainable reference implementation for removing a watermark from **video you own or have permission to edit**.

This version focuses on engineering quality as well as reconstruction quality.

## Design goals

The codebase follows a few deliberate principles:

- **Single Responsibility Principle** — alpha estimation, masks, video I/O, external-process integration, reporting, and orchestration live in separate modules.
- **Dependency Inversion** — external commands and the inpainting backend are accessed through small interfaces/adapters rather than being hard-coded into the image-processing logic.
- **Pure functions where possible** — the mathematically interesting alpha/recovery code is isolated and easy to unit test.
- **Fail fast** — configuration and external dependencies are validated before a long video run begins.
- **Typed data models** — regions, configuration, and quality metrics are represented with dataclasses instead of unstructured dictionaries.
- **Deterministic filenames and explicit outputs** — intermediate responsibilities are predictable and easier to debug.
- **Testable boundaries** — subprocess calls are wrapped so tests can use fakes instead of launching FFmpeg or ProPainter.
- **Separation of policy from mechanism** — thresholds live in `PipelineConfig`; algorithms consume configuration rather than embedding magic values throughout the code.

## Package structure

```text
watermark_remover/
├── __init__.py
├── alpha.py        # alpha estimation, inverse compositing, analytic blending
├── cli.py          # command-line interface only
├── infra.py        # subprocess and ProPainter adapters
├── masks.py        # mask loading and region fallback
├── models.py       # typed configuration/value objects
├── pipeline.py     # application orchestration
├── ports.py        # lightweight interfaces / protocols
├── reporting.py    # CSV quality reporting
└── video.py        # frame extraction, probing, audio muxing

tests/
├── test_alpha.py
├── test_infra.py
├── test_masks.py
├── test_models.py
└── test_reporting.py
```

## Technique

The observed watermark composite is approximately:

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

This becomes unstable near `α = 1`, so the pipeline does not trust analytic recovery everywhere.

Instead:

```text
precise watermark support mask
        ↓
temporal background estimate
        ↓
soft alpha + foreground estimate
        ↓
inverse compositing
        ↓
confidence map
        ├── confident region → analytic recovery
        └── uncertain region → residual mask → ProPainter
```

The intended production architecture is:

```text
SAM 2.1 exact mask propagation
        ↓
alpha-aware analytic recovery
        ↓
small residual uncertainty mask
        ↓
ProPainter temporal inpainting
        ↓
quality metrics + original audio
```

## Why this architecture

Traditional blur/delogo methods damage the entire marked area.

Pure generative inpainting may unnecessarily invent pixels that are already partially visible through a translucent watermark.

The hybrid pipeline preserves more source information:

- semi-transparent edges are mathematically recovered where possible;
- opaque or uncertain pixels are handled by temporal video inpainting;
- external AI tooling is isolated behind adapters.

## Installation

Create a virtual environment, then:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

Install FFmpeg separately and ensure:

```bash
ffmpeg -version
```

works.

Install ProPainter separately according to its upstream project instructions.

## Running

With a SAM 2 / SAM 2.1 mask directory:

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --mask-dir ./sam_masks \
  --fp16
```

With a known fallback region:

```bash
watermark-remove input.mp4 \
  --propainter ./ProPainter \
  --x 1115 --y 540 --w 125 --h 130 \
  --fp16
```

If neither a mask directory nor coordinates are supplied, the fallback region is tuned to the original example's bottom-right mark.

## Tests

Run:

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

The tests deliberately focus on the deterministic core:

- region clamping
- configuration validation
- temporal median
- known inverse-compositing case
- residual-mask threshold behaviour
- analytic blending
- mask resizing/loading
- CSV reporting
- external adapter validation

Heavy integration tests that require FFmpeg, a GPU, SAM 2, or ProPainter should be kept separate from unit tests and marked as integration tests in CI.

## Recommended CI layout

A practical pipeline is:

```text
1. ruff check .
2. mypy watermark_remover
3. pytest --cov=watermark_remover
4. optional GPU integration job
```

Do not make ordinary pull requests depend on a large GPU model download unless your infrastructure is specifically designed for that.

## Extension points

The code is intentionally structured so you can replace components without rewriting the whole pipeline.

### Replace ProPainter

Implement the `VideoInpainter` protocol.

### Replace subprocess execution

Implement `CommandRunner`.

This is useful for tests, remote execution, containers, or job queues.

### Add SAM 2.1 directly

Add a `MaskProvider` abstraction, then implement:

```text
Sam21MaskProvider
RegionMaskProvider
DirectoryMaskProvider
```

The pipeline currently supports the directory and region behaviours directly; extracting them into a formal strategy is the next small refactor if multiple segmentation backends are used concurrently.

## Best-practice improvements still worth adding

For a production-grade service:

- structured logging instead of `print`;
- retry policy for external-process failures;
- explicit cancellation handling;
- disk-space preflight checks;
- streaming/chunked processing instead of loading all frames into RAM;
- scene-level batching;
- GPU capability detection;
- model/version manifest in the report;
- deterministic run metadata;
- resumable intermediate artifacts;
- integration-test fixtures using a tiny synthetic video;
- Docker/Conda lockfiles;
- security review for arbitrary external paths.

## Important limitation

Alpha and watermark colour cannot in general be uniquely recovered from a single composite image.

The current estimator uses temporal statistics and residual magnitude as an engineering approximation. It can preserve more real image information for translucent overlays, but it does not guarantee exact reconstruction of pixels that were never visible.

Use the tool only on material you own or have permission to modify.
