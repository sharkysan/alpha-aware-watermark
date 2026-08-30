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
from .infra import ProPainterAdapter, SubprocessCommandRunner, require_executable
from .masks import load_frame_mask, region_mask
from .models import PipelineConfig, QualityMetrics, Region
from .reporting import write_quality_report
from .video import extract_frames, mux_original_audio, probe_video


class WatermarkRemovalPipeline:
    """Application service orchestrating the full watermark-removal workflow."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.runner = SubprocessCommandRunner()

    def run(self) -> Path:
        self.config.validate()
        require_executable("ffmpeg")

        if not self.config.input_path.exists():
            raise FileNotFoundError(self.config.input_path)

        width, height, _, _ = probe_video(self.config.input_path)
        region = self._resolve_region(width, height)

        inpainter = ProPainterAdapter(
            repo_dir=self.config.propainter_dir,
            runner=self.runner,
            neighbor_length=self.config.neighbor_length,
            ref_stride=self.config.ref_stride,
            resize_ratio=self.config.resize_ratio,
            fp16=self.config.fp16,
        )
        inpainter.validate()

        metrics: list[QualityMetrics] = []

        with tempfile.TemporaryDirectory(prefix="alpha_wm_") as temp:
            temp_dir = Path(temp)
            raw_dir = temp_dir / "frames"
            analytic_dir = temp_dir / "analytic"
            residual_dir = temp_dir / "residual_masks"
            inpainted_dir = temp_dir / "inpainted"
            debug_dir = temp_dir / "debug"

            analytic_dir.mkdir()
            residual_dir.mkdir()
            if self.config.save_debug:
                debug_dir.mkdir()

            frame_paths = extract_frames(
                self.config.input_path,
                raw_dir,
                self.runner,
            )
            frames = [cv2.imread(str(path)) for path in frame_paths]
            if any(frame is None for frame in frames):
                raise RuntimeError("One or more extracted frames could not be read.")

            for index, observed in enumerate(frames):
                window = self._temporal_window(frames, index)
                background = temporal_median(window)
                support = self._support_mask(index, width, height, region)

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

            generated_video = inpainter.inpaint(
                analytic_dir,
                residual_dir,
                inpainted_dir,
            )
            mux_original_audio(
                generated_video,
                self.config.input_path,
                self.config.output_path,
                self.runner,
            )

        write_quality_report(self.config.report_path, metrics)
        return self.config.output_path

    def _resolve_region(self, width: int, height: int) -> Region:
        if self.config.region is not None:
            return self.config.region.clamp(width, height)
        return Region(
            x=int(width * 0.872),
            y=int(height * 0.755),
            width=int(width * 0.095),
            height=int(height * 0.175),
        ).clamp(width, height)

    def _temporal_window(
        self,
        frames: list[np.ndarray],
        index: int,
    ) -> list[np.ndarray]:
        radius = self.config.temporal_radius
        start = max(0, index - radius)
        end = min(len(frames), index + radius + 1)
        return frames[start:end]

    def _support_mask(
        self,
        index: int,
        width: int,
        height: int,
        region: Region,
    ) -> np.ndarray:
        if self.config.mask_dir is not None:
            mask = load_frame_mask(
                self.config.mask_dir,
                index,
                width,
                height,
            )
            if mask is not None:
                return mask
        return region_mask(width, height, region)
