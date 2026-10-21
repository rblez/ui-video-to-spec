"""Paso 7: los entregables se escriben con los valores medidos."""

import json

from ui_video_to_spec import generate
from ui_video_to_spec.types import (
    Animation,
    ColorInfo,
    Element,
    Measurements,
    ShadowInfo,
    VideoMeta,
)


def _fake_measurements() -> Measurements:
    m = Measurements(meta=VideoMeta("demo.mp4", 3.0, 1280, 720, 30.0, False))
    m.palette = [
        ColorInfo("#EEF1F6", (238, 241, 246), 0.62, role="background"),
        ColorInfo("#FFFFFF", (255, 255, 255), 0.25, role="surface"),
        ColorInfo("#2563EB", (37, 99, 235), 0.03, role="primary"),
    ]
    m.elements = [
        Element(id="el-0", kind="screen", bbox=(0, 0, 1280, 720),
                fill_hex="#EEF1F6"),
        Element(id="el-1", kind="card", bbox=(400, 180, 480, 360),
                fill_hex="#FFFFFF", radius_px=16,
                shadow=ShadowInfo(0, 10, 18, "#0F172A", 0.16),
                children=["el-2"], parent_id="el-0", layer=1),
        Element(id="el-2", kind="button", bbox=(490, 430, 300, 56),
                fill_hex="#2563EB", radius_px=10, parent_id="el-1", layer=2,
                text="Comprar ahora"),
    ]
    m.spacing_scale = [8, 16, 24, 40]
    m.animations = [Animation(
        element_id="el-2", property="opacity", duration_ms=600,
        delay_ms=1000, easing=(0.22, 1.0, 0.36, 1.0),
        easing_name="ease-out-quint", from_value=0.0, to_value=1.0, unit="")]
    return m


def test_write_outputs(tmp_path):
    m = _fake_measurements()
    tokens = generate.write_json_outputs(m, tmp_path)
    assert tokens["color"]["primary"]["value"] == "#2563EB"
    data = json.loads((tmp_path / "design-tokens.json").read_text())
    assert data["color"]["background"]["value"] == "#EEF1F6"
    motion = json.loads((tmp_path / "motion.json").read_text())
    anim = motion["animations"][0]
    assert anim["duration_ms"] == 600 and anim["delay_ms"] == 1000
    assert anim["easing"] == [0.22, 1.0, 0.36, 1.0]

    components = generate.write_components(m, tmp_path)
    assert components
    screen = (tmp_path / "components" / "Screen.tsx").read_text()
    assert "#2563EB" in screen and "borderRadius: 16" in screen

    skill = generate.write_skill(m, tokens, tmp_path)
    text = (tmp_path / "SKILL.md").read_text()
    assert "#2563EB" in text and "cubic-bezier" in text

    generate.write_report(m, tmp_path, {"method": "pil-synthetic",
                                        "attempts": 1, "ssim": 0.9,
                                        "threshold": 0.92, "passed": False})
    report = (tmp_path / "report.md").read_text()
    assert "medido" in report
    assert "hover" in report and "Dark mode" in report  # sección no-deducible
