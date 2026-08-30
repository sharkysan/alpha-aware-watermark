from __future__ import annotations

import csv
from pathlib import Path

from .models import QualityMetrics


def write_quality_report(path: Path, metrics: list[QualityMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame",
                "support_pixels",
                "residual_inpaint_pixels",
                "mean_alpha",
                "mean_analytic_confidence",
                "residual_fraction",
            ]
        )
        for item in metrics:
            writer.writerow(
                [
                    item.frame_index,
                    item.support_pixels,
                    item.residual_pixels,
                    f"{item.mean_alpha:.6f}",
                    f"{item.mean_confidence:.6f}",
                    f"{item.residual_fraction:.6f}",
                ]
            )
