from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from .alpha import (
    blend_analytic,
    estimate_alpha_and_foreground,
    inverse_composite,
    residual_inpaint_mask,
    temporal_median,
)
from .checkpoints import CheckpointStore
from .chunks import FrameChunk, iter_scene_chunks
from .disk import ensure_pipeline_disk_space, estimate_disk_space
from .infra import ProPainterAdapter, SubprocessCommandRunner, require_executable
from .mask_providers import DirectoryMaskProvider, MaskProvider, RegionMaskProvider
from .models import PipelineConfig, QualityMetrics, Region
from .motion import aligned_temporal_median
from .ports import CommandRunner
from .progress import PipelineProgress, ProgressReporter, ignore_progress
from .reporting import write_quality_report
from .scenes import Scene, detect_scenes
from .video import extract_frames, mux_original_audio, probe_video


class WatermarkRemovalPipeline:
    """Application service orchestrating the watermark-removal workflow."""

    def __init__(
        self,
        config: PipelineConfig,
        runner: CommandRunner | None = None,
        mask_provider: MaskProvider | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or SubprocessCommandRunner()
        self._mask_provider = mask_provider
        self._progress_reporter = progress_reporter or ignore_progress

    def run(self) -> Path:
        self._report(0.01, "validation", "Validating configuration")
        self.config.validate()
        require_executable("ffmpeg")

        if not self.config.input_path.exists():
            raise FileNotFoundError(self.config.input_path)

        width, height, _, frame_count = probe_video(self.config.input_path)
        self._report(0.05, "preflight", "Checking disk space and processing prerequisites")
        self._preflight_disk_space(width, height, frame_count)
        mask_provider = self._mask_provider or self._build_mask_provider(width, height)

        inpainter = ProPainterAdapter(
            repo_dir=self.config.propainter_dir,
            runner=self.runner,
            neighbor_length=self.config.neighbor_length,
            ref_stride=self.config.ref_stride,
            resize_ratio=self.config.resize_ratio,
            fp16=self.config.fp16,
            python_executable=self.config.propainter_python,
        )
        inpainter.validate()

        metrics: list[QualityMetrics] = []
        checkpoint_store = CheckpointStore(self.config) if self.config.resume else None
        if checkpoint_store is not None:
            checkpoint_store.prepare()
            self._report(0.07, "checkpoint", "Prepared resumable chunk checkpoints")

        with tempfile.TemporaryDirectory(prefix="alpha_wm_") as temp:
            temp_dir = Path(temp)
            raw_dir = temp_dir / "frames"
            inpainted_dir = temp_dir / "inpainted"

            if checkpoint_store is None:
                analytic_dir = temp_dir / "analytic"
                residual_dir = temp_dir / "residual_masks"
                debug_dir = temp_dir / "debug"
                analytic_dir.mkdir()
                residual_dir.mkdir()
                if self.config.save_debug:
                    debug_dir.mkdir()
            else:
                analytic_dir = checkpoint_store.analytic_dir
                residual_dir = checkpoint_store.residual_dir
                debug_dir = checkpoint_store.debug_dir

            self._report(0.10, "extracting", "Extracting source frames")
            frame_paths = extract_frames(
                self.config.input_path,
                raw_dir,
                self.runner,
            )
            self._report(0.18, "scenes", "Detecting scene boundaries")
            scenes = detect_scenes(
                frame_paths,
                threshold=self.config.scene_threshold,
                min_scene_length=self.config.min_scene_length,
            )

            processed_frames = 0
            total_frames = max(len(frame_paths), 1)
            for scene in scenes:
                for chunk in iter_scene_chunks(
                    scene,
                    chunk_size=self.config.chunk_size,
                    temporal_radius=self.config.temporal_radius,
                ):
                    chunk_metrics = (
                        checkpoint_store.load_chunk(chunk)
                        if checkpoint_store is not None
                        else None
                    )
                    if chunk_metrics is None:
                        loaded = self._load_chunk(frame_paths, chunk)
                        chunk_metrics = self._process_chunk(
                            loaded=loaded,
                            scene=scene,
                            chunk=chunk,
                            mask_provider=mask_provider,
                            frame_width=width,
                            frame_height=height,
                            analytic_dir=analytic_dir,
                            residual_dir=residual_dir,
                            debug_dir=debug_dir,
                        )
                        if checkpoint_store is not None:
                            checkpoint_store.save_chunk(chunk, chunk_metrics)
                    else:
                        self._report(
                            0.20 + 0.55 * min(processed_frames / total_frames, 1.0),
                            "resuming",
                            (
                                "Reusing checkpointed frames "
                                f"{chunk.process_start}-{chunk.process_end - 1}"
                            ),
                        )

                    metrics.extend(chunk_metrics)
                    processed_frames += chunk.process_end - chunk.process_start
                    processing_fraction = min(processed_frames / total_frames, 1.0)
                    self._report(
                        0.20 + 0.55 * processing_fraction,
                        "processing",
                        f"Processed {min(processed_frames, total_frames)}/{total_frames} frames",
                    )

            self._report(0.80, "inpainting", "Running ProPainter residual inpainting")
            generated_video = inpainter.inpaint(
                analytic_dir,
                residual_dir,
                inpainted_dir,
            )
            self._report(0.92, "muxing", "Restoring original audio and writing output video")
            mux_original_audio(
                generated_video,
                self.config.input_path,
                self.config.output_path,
                self.runner,
            )

        self._report(0.98, "reporting", "Writing quality report")
        write_quality_report(self.config.report_path, metrics)
        if checkpoint_store is not None:
            checkpoint_store.cleanup()
        self._report(1.0, "complete", "Processing completed successfully")
        return self.config.output_path

    def _report(self, fraction: float, stage: str, message: str) -> None:
        self._progress_reporter(PipelineProgress(fraction, stage, message))

    def _preflight_disk_space(self, width: int, height: int, frame_count: int) -> None:
        estimate = estimate_disk_space(
            width=width,
            height=height,
            frame_count=frame_count,
            input_size_bytes=self.config.input_path.stat().st_size,
            save_debug=self.config.save_debug,
        )
        ensure_pipeline_disk_space(
            Path(tempfile.gettempdir()),
            self.config.output_path.parent,
            estimate,
        )

    def _build_mask_provider(self, width: int, height: int) -> MaskProvider:
        region = self._resolve_region(width, height)
        fallback = RegionMaskProvider(region)
        if self.config.mask_dir is not None:
            return DirectoryMaskProvider(self.config.mask_dir, fallback=fallback)
        return fallback

    def _resolve_region(self, width: int, height: int) -> Region:
        if self.config.region is not None:
            return self.config.region.clamp(width, height)
        return Region(
            x=int(width * 0.872),
            y=int(height * 0.755),
            width=int(width * 0.095),
            height=int(height * 0.175),
        ).clamp(width, height)

    @staticmethod
    def _load_chunk(
        frame_paths: list[Path],
        chunk: FrameChunk,
    ) -> dict[int, np.ndarray]:
        loaded: dict[int, np.ndarray] = {}
        for index in range(chunk.load_start, chunk.load_end):
            frame = cv2.imread(str(frame_paths[index]))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_paths[index]}")
            loaded[index] = frame
        return loaded

    def _process_chunk(
        self,
        loaded: dict[int, np.ndarray],
        scene: Scene,
        chunk: FrameChunk,
        mask_provider: MaskProvider,
        frame_width: int,
        frame_height: int,
        analytic_dir: Path,
        residual_dir: Path,
        debug_dir: Path,
    ) -> list[QualityMetrics]:
        metrics: list[QualityMetrics] = []

        for index in range(chunk.process_start, chunk.process_end):
            observed = loaded[index]
            neighbor_indices = self._neighbor_indices(index, scene)
            neighbors = [loaded[i] for i in neighbor_indices if i != index]
            background = self._estimate_background(observed, neighbors)
            support = mask_provider.get_mask(
                index,
                frame_width,
                frame_height,
            )

            alpha, foreground = estimate_alpha_and_foreground(
                observed,
                background,
                support,
            )
            recovered, confidence = inverse_composite(
                observed,
                alpha,
                foreground,
            )
            analytic = blend_analytic(
                observed,
                recovered,
                alpha,
                confidence,
            )
            residual = residual_inpaint_mask(
                alpha=alpha,
                confidence=confidence,
                support_mask=support,
                alpha_threshold=self.config.alpha_inpaint_threshold,
                confidence_min=self.config.analytic_confidence_min,
                dilate=self.config.residual_dilate,
            )

            cv2.imwrite(str(analytic_dir / f"{index:05d}.png"), analytic)
            cv2.imwrite(str(residual_dir / f"{index:05d}.png"), residual)

            if self.config.save_debug:
                cv2.imwrite(
                    str(debug_dir / f"{index:05d}_alpha.png"),
                    (np.clip(alpha, 0, 1) * 255).astype(np.uint8),
                )
                cv2.imwrite(
                    str(debug_dir / f"{index:05d}_confidence.png"),
                    (np.clip(confidence, 0, 1) * 255).astype(np.uint8),
                )

            support_pixels = int((support > 0).sum())
            residual_pixels = int((residual > 0).sum())
            metrics.append(
                QualityMetrics(
                    frame_index=index,
                    support_pixels=support_pixels,
                    residual_pixels=residual_pixels,
                    mean_alpha=float(alpha[support > 0].mean())
                    if support_pixels
                    else 0.0,
                    mean_confidence=float(confidence[support > 0].mean())
                    if support_pixels
                    else 0.0,
                )
            )

        return metrics

    def _neighbor_indices(self, index: int, scene: Scene) -> list[int]:
        radius = self.config.temporal_radius
        start = max(scene.start, index - radius)
        end = min(scene.end, index + radius + 1)
        return list(range(start, end))

    def _estimate_background(
        self,
        observed: np.ndarray,
        neighbors: list[np.ndarray],
    ) -> np.ndarray:
        if self.config.motion_compensation:
            return aligned_temporal_median(observed, neighbors)
        return temporal_median([observed, *neighbors])
