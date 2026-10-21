"""Paso 8 — Loop de verificación visual contra el video original.

Renderiza los elementos medidos y compara el resultado con el frame de
referencia del video:

- **Render**: Playwright si está instalado (extra ``[verify]``) sobre un
  HTML equivalente; si no, render sintético con PIL (rectángulos
  redondeados con los colores/radios/sombras medidos). La degradación se
  anota en el resumen, nunca se oculta.
- **Comparación**: SSIM (implementación propia con ventana gaussiana) +
  diferencia media de color.
- **Iteración**: hasta ``max_attempts`` o superar ``ssim_threshold``.
  Entre intentos se refinan los rellenos muestreando la mediana del color
  real dentro de cada bbox en el frame de referencia.
- Los diffs visuales (diferencia amplificada) quedan en ``out/verify/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .measure_static import load_rgb, rgb_to_hex
from .types import Config, Measurements


def ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """SSIM medio entre dos imágenes RGB del mismo tamaño."""
    a = np.asarray(Image.fromarray(img_a).convert("L"), dtype=np.float64)
    b = np.asarray(Image.fromarray(img_b).convert("L"), dtype=np.float64)
    if a.shape != b.shape:
        b = np.asarray(
            Image.fromarray(img_b).convert("L").resize(
                (a.shape[1], a.shape[0])), dtype=np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def render_synthetic(m: Measurements) -> Image.Image:
    """Dibuja los elementos medidos con PIL (fondo, sombras, radios)."""
    w, h = m.meta.width, m.meta.height
    bg = m.palette[0].hex if m.palette else "#FFFFFF"
    img = Image.new("RGB", (w, h), bg)
    for el in sorted((e for e in m.elements if e.id != "el-0"),
                     key=lambda e: e.layer):
        x, y, ew, eh = el.bbox
        if el.shadow:
            s = el.shadow
            layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            d.rounded_rectangle(
                [x + s.offset_x, y + s.offset_y,
                 x + ew + s.offset_x, y + eh + s.offset_y],
                radius=el.radius_px, fill=(0, 0, 0, int(255 * s.opacity)))
            img.paste(Image.alpha_composite(
                img.convert("RGBA"),
                layer.filter(ImageFilter.GaussianBlur(max(1, s.blur / 2))),
            ).convert("RGB"), (0, 0))
        if el.fill_hex:
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([x, y, x + ew, y + eh],
                                radius=el.radius_px, fill=el.fill_hex)
    return img


def render_playwright(m: Measurements, out_html: Path, out_png: Path) -> bool:
    """Render con Playwright si está disponible. False si degrada."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        return False
    divs = []
    for el in sorted((e for e in m.elements if e.id != "el-0"),
                     key=lambda e: e.layer):
        x, y, w, h = el.bbox
        style = (f"position:absolute;left:{x}px;top:{y}px;width:{w}px;"
                 f"height:{h}px;background:{el.fill_hex or 'transparent'};"
                 f"border-radius:{el.radius_px}px;")
        if el.shadow:
            style += f"box-shadow:{el.shadow.to_css()};"
        divs.append(f'<div style="{style}"></div>')
    bg = m.palette[0].hex if m.palette else "#FFFFFF"
    html = (f'<html><body style="margin:0;background:{bg};'
            f'width:{m.meta.width}px;height:{m.meta.height}px;'
            f'position:relative">{"".join(divs)}</body></html>')
    out_html.write_text(html, encoding="utf-8")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": m.meta.width, "height": m.meta.height})
            page.goto(out_html.as_uri())
            page.screenshot(path=str(out_png))
            browser.close()
        return True
    except Exception:
        return False


def _refine_fills(m: Measurements, reference: np.ndarray) -> None:
    """Ajusta cada relleno a la mediana real dentro de su bbox."""
    for el in m.elements:
        if el.id == "el-0":
            continue
        x, y, w, h = el.bbox
        crop = reference[max(0, y):y + h, max(0, x):x + w]
        if crop.size:
            el.fill_hex = rgb_to_hex(np.median(
                crop.reshape(-1, 3), axis=0))


def verify(m: Measurements, out_dir: Path, cfg: Config) -> dict:
    """Loop de verificación. Devuelve el resumen (también a verify/summary.json)."""
    vdir = out_dir / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"method": "none", "attempts": 0, "ssim": 0.0,
                     "threshold": cfg.ssim_threshold, "passed": False,
                     "color_diff": None}
    if m.reference_frame is None or m.meta is None:
        summary["note"] = "Sin frame de referencia; verificación omitida."
        (vdir / "summary.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")
        return summary
    reference = load_rgb(m.reference_frame)
    use_pw = (vdir / "render.html").exists() or True  # se intenta siempre
    method = "pil-synthetic"
    best = 0.0
    for attempt in range(1, cfg.max_attempts + 1):
        shot = vdir / f"render_attempt_{attempt}.png"
        rendered = None
        if use_pw and render_playwright(m, vdir / "render.html", shot):
            method = "playwright"
            rendered = load_rgb(shot)
        else:
            img = render_synthetic(m)
            img.save(shot)
            rendered = np.asarray(img)
        score = ssim(reference, rendered)
        color_diff = float(np.mean(np.abs(
            reference.astype(np.float32) - rendered.astype(np.float32))))
        diff = Image.fromarray(np.clip(np.abs(
            reference.astype(np.int16) - rendered.astype(np.int16)) * 3,
            0, 255).astype(np.uint8))
        diff.save(vdir / f"diff_attempt_{attempt}.png")
        best = max(best, score)
        summary.update({"method": method, "attempts": attempt,
                        "ssim": round(score, 4), "color_diff": round(color_diff, 2),
                        "passed": score >= cfg.ssim_threshold})
        if score >= cfg.ssim_threshold:
            break
        if attempt < cfg.max_attempts:
            _refine_fills(m, reference)
    if method != "playwright":
        summary["note"] = ("Playwright no disponible; se usó render sintético "
                           "con PIL (degradación anotada).")
    (vdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
