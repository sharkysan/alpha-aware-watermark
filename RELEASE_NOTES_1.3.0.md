# Alpha-Aware Watermark Remover 1.3.0

Version 1.3.0 improves long-running reliability and the local Gradio workflow while keeping the existing alpha-aware, scene-aware temporal pipeline intact.

## Highlights

### Resumable chunk checkpoints

Long-running jobs can now persist completed analytic/residual chunks and reuse them after an interruption.

```bash
watermark-remove input.mp4 \
  --propainter /path/to/ProPainter \
  --resume
```

Use `--checkpoint-dir` to place checkpoint state on a separate filesystem or faster local disk.

Checkpoint reuse is protected by a fingerprint derived from the input identity, masks/region and processing settings. Incomplete or incompatible checkpoint state is rejected automatically. Per-input checkpoint data is removed after a fully successful run.

See [`CHECKPOINTS.md`](CHECKPOINTS.md) for details.

### Interactive watermark-region selection

The Gradio UI can now load a preview frame and define a rectangular watermark region by clicking two opposite corners. The selected region is written back into the same `PipelineConfig` used by the CLI, so UI and command-line behaviour remain aligned.

### UI preflight readiness panel

The local UI now exposes preflight checks before starting expensive work. It reports input, ProPainter and disk-readiness issues earlier, including correct accounting when scratch/output paths share a filesystem.

### Processing progress reporting

The pipeline now emits structured progress updates and the Gradio UI surfaces them during processing. Progress stages are kept outside the core algorithms through a small reporter abstraction, preserving testability and CLI/UI separation.

### Disk-preflight hardening

Disk-space checks now account for shared filesystems correctly instead of treating scratch and output requirements as independent when they consume the same capacity.

### Updated UI documentation

The README now includes a Gradio UI overview screenshot and the UI documentation has been expanded around interactive selection, preflight and progress behaviour.

## Quality and testing

The release adds or expands tests for:

- checkpoint fingerprinting and invalidation;
- safe checkpoint cleanup;
- rejection of incomplete checkpoint outputs;
- resumable chunk reuse;
- shared-filesystem disk accounting;
- interactive region selection;
- UI preflight reporting;
- progress model and UI reporter delegation.

Existing CI/security gates remain in place: Ruff, mypy, pytest/coverage, Bandit, pip-audit, package build/metadata validation and CodeQL.

## Upgrade notes

There are no intentional breaking changes to the existing CLI workflow.

New CLI options:

```text
--resume
--checkpoint-dir PATH
```

`--checkpoint-dir` requires `--resume`.

Existing region masks, per-frame masks and UI workflows remain supported.

## Known limitations

- Checkpoints currently cover the analytic/residual chunk phase. Source-frame extraction and the final ProPainter pass are repeated after restart.
- ProPainter remains an external dependency with its own installation and licence terms.
- Direct SAM 2.1 propagation is still not integrated as an in-package mask provider.
- The Gradio UI is intended primarily for trusted local use; review [`UI.md`](UI.md) before network exposure.
- Scratch storage can still be substantial for long or high-resolution videos.
