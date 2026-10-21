"""Paso 1: extracción adaptativa de frames."""

import json
from pathlib import Path

from ui_video_to_spec.extract_frames import extract_frames, probe_video
from ui_video_to_spec.types import Config


def test_probe_video(example_video):
    meta = probe_video(example_video)
    assert meta.width == 640 and meta.height == 360
    assert 2.0 < meta.duration_s < 2.8
    assert meta.has_audio is False


def test_extract_detects_motion_window(example_video, tmp_path):
    cfg = Config(fps_static=2, fps_motion=24)
    meta, frames, windows = extract_frames(example_video, tmp_path / "w", cfg)
    assert frames, "debe extraer frames"
    assert all(Path(f.path).is_file() for f in frames)
    assert any(f.kind == "static" for f in frames)
    # La animación del botón ocurre en t≈1.0–1.6s: debe haber ventana ahí.
    assert windows, "debe detectar la ventana de movimiento del botón"
    assert any(a <= 1.6 and b >= 1.0 for a, b in windows)
    assert any(f.kind == "motion" for f in frames)
    manifest = json.loads((tmp_path / "w" / "manifest.json").read_text())
    assert manifest["frames"] and manifest["motion_windows"]
