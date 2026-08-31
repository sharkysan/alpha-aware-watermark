from __future__ import annotations

import argparse
from pathlib import Path

from .models import PipelineConfig, Region
from .pipeline import WatermarkRemovalPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpha-aware temporal watermark remover."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--propainter", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--mask-dir", type=Path)

    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--h", type=int)

    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--scene-threshold", type=float, default=0.62)
    parser.add_argument("--min-scene-length", type=int, default=10)
    parser.add_argument(
        "--no-motion-compensation",
        action="store_true",
        help="Disable optical-flow alignment before temporal background estimation.",
    )
    parser.add_argument("--alpha-inpaint", type=float, default=0.55)
    parser.add_argument("--analytic-confidence-min", type=float, default=0.28)
    parser.add_argument("--residual-dilate", type=int, default=3)

    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--resize-ratio", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Persist completed analytic chunks and reuse them after an interrupted run.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Checkpoint directory; requires --resume.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output = args.output or args.input.with_name(
        args.input.stem + "_alpha_clean.mp4"
    )
    report = args.input.with_name(args.input.stem + "_alpha_quality.csv")

    region = None
    coords = [args.x, args.y, args.w, args.h]
    if any(value is not None for value in coords):
        if not all(value is not None for value in coords):
            raise SystemExit("--x, --y, --w and --h must be supplied together.")
        region = Region(args.x, args.y, args.w, args.h)

    config = PipelineConfig(
        input_path=args.input,
        propainter_dir=args.propainter,
        output_path=output,
        report_path=report,
        mask_dir=args.mask_dir,
        region=region,
        temporal_radius=args.temporal_radius,
        chunk_size=args.chunk_size,
        scene_threshold=args.scene_threshold,
        min_scene_length=args.min_scene_length,
        motion_compensation=not args.no_motion_compensation,
        alpha_inpaint_threshold=args.alpha_inpaint,
        analytic_confidence_min=args.analytic_confidence_min,
        residual_dilate=args.residual_dilate,
        neighbor_length=args.neighbor_length,
        ref_stride=args.ref_stride,
        resize_ratio=args.resize_ratio,
        fp16=args.fp16,
        save_debug=args.save_debug,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )

    result = WatermarkRemovalPipeline(config).run()
    print(f"Done: {result}")
    print(f"Quality report: {report}")


if __name__ == "__main__":
    main()
