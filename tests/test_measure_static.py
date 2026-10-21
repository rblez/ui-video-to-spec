"""Paso 3: paleta, elementos, radios — contra valores conocidos."""

import numpy as np
from PIL import Image, ImageDraw

from ui_video_to_spec.measure_static import (
    extract_palette,
    find_elements,
    measure_static,
)
from ui_video_to_spec.extract_frames import extract_frames
from ui_video_to_spec.types import Config


def _hex_close(hex_a: str, hex_b: str, tol: int = 14) -> bool:
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _synthetic_image() -> np.ndarray:
    img = Image.new("RGB", (640, 360), (238, 241, 246))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([200, 90, 440, 270], radius=16, fill=(255, 255, 255))
    d.rounded_rectangle([245, 215, 395, 243], radius=10, fill=(37, 99, 235))
    return np.asarray(img)


def test_palette_finds_known_colors():
    palette = extract_palette(_synthetic_image(), 8)
    hexes = [c.hex for c in palette]
    assert _hex_close(hexes[0], "#EEF1F6")           # fondo dominante
    assert any(_hex_close(h, "#FFFFFF") for h in hexes)
    assert any(_hex_close(h, "#2563EB") for h in hexes)


def test_find_elements_geometry():
    img = _synthetic_image()
    palette = extract_palette(img, 8)
    elements = find_elements(img, palette, min_area=900)
    card = next((e for e in elements if e.fill_hex and
                 _hex_close(e.fill_hex, "#FFFFFF")), None)
    assert card is not None
    x, y, w, h = card.bbox
    assert abs(x - 200) <= 6 and abs(y - 90) <= 6
    assert abs(w - 240) <= 8 and abs(h - 180) <= 8
    assert 8 <= card.radius_px <= 24  # radio real: 16
    button = next((e for e in elements if e.fill_hex and
                   _hex_close(e.fill_hex, "#2563EB")), None)
    assert button is not None and button.parent_id == card.id


def test_measure_static_on_video(example_video, tmp_path, expected):
    cfg = Config(fps_static=2, fps_motion=24)
    meta, frames, _ = extract_frames(example_video, tmp_path / "w", cfg)
    m = measure_static(frames, meta, cfg)
    assert m.palette and _hex_close(m.palette[0].hex, expected["background"])
    fills = [e.fill_hex for e in m.elements if e.fill_hex]
    assert any(_hex_close(f, "#FFFFFF") for f in fills)
