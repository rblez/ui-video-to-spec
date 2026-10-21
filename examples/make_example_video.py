#!/usr/bin/env python3
"""Genera un video de UI sintético con valores CONOCIDOS (ground truth).

Sirve para probar el pipeline: si las mediciones son honestas, deben salir
aproximadamente estos valores (la compresión H.264 mueve un poco los
colores; por eso la paleta se mide con medianas):

- Fondo:        #EEF1F6
- Card:         #FFFFFF, radio 16px, bbox (400,180) 480×360 a 1280×720,
                sombra suave (offset y +10, blur ~18)
- Botón:        #2563EB, radio 10px, dentro de la card
- Animación:    el botón aparece con opacity 0→1 y translateY 24→0,
                delay 1000ms, duración 600ms,
                easing cubic-bezier(0.22, 1, 0.36, 1)

Uso:
    python make_example_video.py [--out ui-demo.mp4] [--width 1280]
                                 [--height 720] [--duration 3.0] [--fps 30]

Escribe además ``expected.json`` junto al video con la verdad de terreno.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (238, 241, 246)        # #EEF1F6
CARD = (255, 255, 255)      # #FFFFFF
BTN = (37, 99, 235)         # #2563EB
TEXT = (17, 24, 39)         # #111827
SUBTEXT = (107, 114, 128)   # #6B7280

CARD_RADIUS = 16
BTN_RADIUS = 10
ANIM_DELAY_MS = 1000
ANIM_DURATION_MS = 600
EASING = (0.22, 1.0, 0.36, 1.0)
SLIDE_PX = 24


def bezier_y(x: float, p=EASING) -> float:
    """Evalúa y(x) del cubic-bezier de la animación (bisección)."""
    x1, y1, x2, y2 = p
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(24):
        u = (lo + hi) / 2
        xu = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u * u * x2 + u ** 3
        if xu < x:
            lo = u
        else:
            hi = u
    u = (lo + hi) / 2
    return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u * u * y2 + u ** 3


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in ("/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/"):
        path = Path(base) / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def geometry(width: int, height: int) -> dict:
    """Posiciones proporcionales a la resolución (para tests pequeños)."""
    sx, sy = width / 1280, height / 720
    card = (int(400 * sx), int(180 * sy), int(880 * sx), int(540 * sy))
    btn = (int(490 * sx), int(430 * sy), int(790 * sx), int(486 * sy))
    return {"card": card, "button": btn, "sx": sx, "sy": sy}


def draw_frame(t_ms: float, width: int, height: int) -> Image.Image:
    geo = geometry(width, height)
    sx, sy = geo["sx"], geo["sy"]
    img = Image.new("RGB", (width, height), BG)
    cx0, cy0, cx1, cy1 = geo["card"]
    # Sombra de la card: capa negra difuminada desplazada hacia abajo.
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([cx0, cy0 + int(10 * sy), cx1, cy1 + int(10 * sy)],
                         radius=CARD_RADIUS, fill=(15, 23, 42, 70))
    img = Image.alpha_composite(
        img.convert("RGBA"),
        shadow.filter(ImageFilter.GaussianBlur(9))).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=CARD_RADIUS, fill=CARD)
    # Textos de la card.
    d.text((cx0 + int(40 * sx), cy0 + int(36 * sy)), "Universal Boutique",
           font=_font(max(12, int(34 * sy)), bold=True), fill=TEXT)
    d.text((cx0 + int(40 * sx), cy0 + int(88 * sy)),
           "Ropa y accesorios — pedidos por WhatsApp",
           font=_font(max(9, int(19 * sy))), fill=SUBTEXT)
    d.text((cx0 + int(40 * sx), cy0 + int(128 * sy)), "$24.99",
           font=_font(max(11, int(26 * sy)), bold=True), fill=TEXT)
    # Botón animado: fade + slide-up con el easing conocido.
    if t_ms >= ANIM_DELAY_MS:
        p = bezier_y(min(1.0, (t_ms - ANIM_DELAY_MS) / ANIM_DURATION_MS))
    else:
        p = 0.0
    if p > 0:
        bx0, by0, bx1, by1 = geo["button"]
        dy = int((1 - p) * SLIDE_PX * sy)
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.rounded_rectangle([bx0, by0 + dy, bx1, by1 + dy],
                             radius=BTN_RADIUS, fill=BTN + (255,))
        label = "Comprar ahora"
        font = _font(max(9, int(20 * sy)), bold=True)
        tw = ld.textlength(label, font=font)
        ld.text((bx0 + (bx1 - bx0 - tw) / 2, by0 + dy + int(16 * sy)),
                label, font=font, fill=(255, 255, 255, 255))
        alpha = layer.getchannel("A").point(lambda a: int(a * p))
        layer.putalpha(alpha)
        base = img.convert("RGBA")
        base.alpha_composite(layer)
        img = base.convert("RGB")
    return img


def make_video(out: Path, width=1280, height=720, duration=3.0, fps=30) -> dict:
    frames_dir = Path(tempfile.mkdtemp(prefix="ui2spec-frames-"))
    n = int(duration * fps)
    for i in range(n):
        t_ms = i / fps * 1000
        draw_frame(t_ms, width, height).save(frames_dir / f"frame_{i:04d}.png")
    cmd = [
        "ffmpeg", "-v", "error", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "30",
        "-movflags", "+faststart", "-y", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg falló: {res.stderr[:400]}")
    geo = geometry(width, height)
    cx0, cy0, cx1, cy1 = geo["card"]
    bx0, by0, bx1, by1 = geo["button"]
    return {
        "resolution": [width, height],
        "background": "#EEF1F6",
        "card": {"fill": "#FFFFFF", "radius_px": CARD_RADIUS,
                 "bbox": [cx0, cy0, cx1 - cx0, cy1 - cy0],
                 "shadow": {"offset_y": 10, "blur_approx": 18}},
        "button": {"fill": "#2563EB", "radius_px": BTN_RADIUS,
                   "bbox": [bx0, by0, bx1 - bx0, by1 - by0]},
        "animation": {
            "target": "button",
            "properties": {
                "opacity": {"from": 0.0, "to": 1.0},
                "translateY": {"from": float(SLIDE_PX), "to": 0.0, "unit": "px"},
            },
            "duration_ms": ANIM_DURATION_MS,
            "delay_ms": ANIM_DELAY_MS,
            "easing": list(EASING),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "ui-demo.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    expected = make_video(out, args.width, args.height, args.duration, args.fps)
    exp_path = out.with_name("expected.json")
    exp_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"Video: {out} · Verdad de terreno: {exp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
