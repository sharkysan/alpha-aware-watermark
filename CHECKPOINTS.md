# Resumable processing checkpoints

Long videos can opt into persistent chunk checkpoints with `--resume`.

```bash
watermark-remove input.mp4 \
  --propainter /path/to/ProPainter \
  --resume
```

Completed analytic/residual chunks are written beneath:

```text
<output-directory>/.alpha_wm_checkpoints/<input-stem>/
```

If a run is interrupted, launching the same command again re-extracts the source frames but skips analytic processing for chunks whose checkpoint records and output files are complete.

Use `--checkpoint-dir` to choose a different checkpoint **parent** directory:

```bash
watermark-remove input.mp4 \
  --propainter /path/to/ProPainter \
  --resume \
  --checkpoint-dir /fast/local/checkpoints
```

The pipeline creates a dedicated `<input-stem>` child inside that parent. It never removes the parent directory itself.

## Safety and invalidation

Checkpoint reuse is guarded by a fingerprint containing:

- source video path, size and modification timestamp;
- mask-directory file identities when masks are used;
- selected region;
- scene/chunk/temporal settings;
- motion-compensation setting;
- alpha/confidence/residual-mask settings;
- debug-output mode;
- an internal checkpoint algorithm version.

When the fingerprint changes, stale checkpoint state is discarded before processing begins.

A chunk is reusable only when its completion record is valid and every expected analytic and residual PNG exists and is non-empty. Partially written chunks are processed again.

## Lifecycle

Checkpoint data is retained when processing fails or is interrupted. After the output video and quality report are written successfully, the per-input checkpoint directory is removed automatically.

The current implementation checkpoints the analytic/residual recovery phase. Source-frame extraction and the final ProPainter pass are repeated after a restart.
