from pathlib import Path

from watermark_remover.models import QualityMetrics
from watermark_remover.reporting import write_quality_report


def test_report_writes_header_and_data(tmp_path: Path):
    path = tmp_path/"report.csv"
    write_quality_report(
        path,
        [QualityMetrics(3, 100, 25, 0.4, 0.8)],
    )
    text = path.read_text(encoding="utf-8")
    assert "residual_fraction" in text
    assert "3,100,25" in text
