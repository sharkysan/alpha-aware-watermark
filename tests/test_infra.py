from pathlib import Path

from watermark_remover.infra import ProPainterAdapter


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, args, cwd=None):
        self.calls.append((args, cwd))


def test_propainter_builds_expected_command(tmp_path: Path):
    repo = tmp_path/"ProPainter"
    repo.mkdir()
    (repo/"inference_propainter.py").write_text("# stub", encoding="utf-8")

    runner = FakeRunner()
    adapter = ProPainterAdapter(
        repo_dir=repo,
        runner=runner,
        neighbor_length=6,
        ref_stride=14,
        resize_ratio=0.75,
        fp16=True,
    )

    adapter.validate()
    assert adapter.inference_script.exists()
