"""Fixtures: un video de ejemplo pequeño y rápido con verdad de terreno."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import make_example_video  # noqa: E402


@pytest.fixture(scope="session")
def example_video(tmp_path_factory) -> Path:
    """Video sintético 640×360, 2.4 s, 24 fps (rápido para tests)."""
    out = tmp_path_factory.mktemp("video") / "ui-demo-small.mp4"
    make_example_video.make_video(out, width=640, height=360,
                                  duration=2.4, fps=24)
    return out


@pytest.fixture(scope="session")
def expected() -> dict:
    return {
        "background": "#EEF1F6",
        "card": {"fill": "#FFFFFF", "radius_px": 16},
        "button": {"fill": "#2563EB", "radius_px": 10},
        "animation": {"duration_ms": 600, "delay_ms": 1000,
                      "easing": [0.22, 1.0, 0.36, 1.0]},
    }
