"""End-to-end: CLI completo sobre el video de ejemplo."""

import json
from pathlib import Path

from ui_video_to_spec.cli import run


def test_end_to_end(example_video, tmp_path):
    out = tmp_path / "out"
    code = run([str(example_video), "--out", str(out),
                "--fps-motion", "24", "--fps-static", "2"])
    assert code == 0
    for name in ("design-tokens.json", "motion.json", "SKILL.md", "report.md"):
        assert (out / name).is_file(), name
    assert (out / "components").is_dir()
    assert (out / "verify" / "summary.json").is_file()

    tokens = json.loads((out / "design-tokens.json").read_text())
    values = [t["value"] for t in tokens["color"].values()]

    def close(hex_a: str, hex_b: str, tol: int = 16) -> bool:
        a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    # El fondo medido debe acercarse al real (#EEF1F6) pese a la compresión.
    assert any(close(v, "#EEF1F6") for v in values)

    motion = json.loads((out / "motion.json").read_text())
    assert isinstance(motion["animations"], list)

    report = (out / "report.md").read_text()
    assert "Lo que NO se puede deducir" in report
