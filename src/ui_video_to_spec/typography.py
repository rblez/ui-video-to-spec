"""Paso 4 — Tipografía: OCR + estimación de métricas (simple, honesto).

- OCR con pytesseract si está instalado (extra ``[ocr]`` y binario
  ``tesseract`` en el sistema). Si no, degrada: sin estilos de texto y una
  nota en el report. Nunca inventa texto.
- Tamaño: la caja de una línea de OCR ≈ 1.25 × el tamaño de fuente, así que
  ``font_size ≈ alto / 1.25`` (inferido).
- Peso: grosor de trazo estimado del recorte binarizado
  (``trazo ≈ 2·área/perímetro`` de los componentes) → 400/500/700.
- Fuentes parecidas: tabla heurística interna. Sin medición real de
  glifos esto es **adivinado** y así se etiqueta en el report.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .types import INFERIDO, TextStyleInfo

# Candidatas sans humanistas/grotescas: cubren la mayoría de UIs reales.
_SANS_CANDIDATES = [
    {"name": "Inter", "score": 0.42},
    {"name": "Roboto", "score": 0.31},
    {"name": "Open Sans", "score": 0.27},
]
_SERIF_CANDIDATES = [
    {"name": "Source Serif 4", "score": 0.35},
    {"name": "Georgia (web-safe)", "score": 0.30},
    {"name": "Lora", "score": 0.25},
]


def _stroke_weight(crop_gray: np.ndarray, font_size: float) -> int:
    """Estima el peso (400/500/700) desde el grosor medio del trazo."""
    if crop_gray.size == 0 or font_size <= 0:
        return 400
    binary = (crop_gray < 128).astype(np.uint8)
    ink = binary.sum()
    if ink == 0:
        return 400
    # Perímetro aproximado: píxeles de tinta con algún vecino vacío.
    padded = np.pad(binary, 1)
    neighbors = sum(
        np.roll(padded, d, axis=a)[1:-1, 1:-1]
        for d, a in ((1, 0), (-1, 0), (1, 1), (-1, 1))
    )
    perimeter = max(1, int(((binary == 1) & (neighbors < 4)).sum()))
    stroke = 2.0 * ink / perimeter
    ratio = stroke / font_size
    if ratio < 0.11:
        return 300
    if ratio < 0.16:
        return 400
    if ratio < 0.21:
        return 500
    return 700


def analyze_typography(image_path: str) -> tuple[list[TextStyleInfo], str | None]:
    """Devuelve (estilos, nota). Nota != None cuando el paso degrada."""
    try:
        import pytesseract  # type: ignore  # extra [ocr]
    except ImportError:
        return [], ("pytesseract no está instalado (extra [ocr]); "
                    "tipografía no medida.")
    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            data = pytesseract.image_to_data(
                rgb, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        return [], f"OCR no disponible en este entorno: {exc}"

    # Agrupa palabras por (bloque, línea) del OCR.
    lines: dict[tuple, dict] = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = float(data["conf"][i]) if str(data["conf"][i]) not in ("-1", "") else -1
        if not text or conf < 30:
            continue
        key = (data["block_num"][i], data["line_num"][i])
        box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        line = lines.setdefault(key, {"words": [], "boxes": []})
        line["words"].append(text)
        line["boxes"].append(box)

    gray = np.asarray(rgb.convert("L"))
    styles: list[TextStyleInfo] = []
    for line in lines.values():
        boxes = line["boxes"]
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        h = max(1, y1 - y0)
        font_size = h / 1.25
        crop = gray[y0:y1, x0:x1]
        weight = _stroke_weight(crop, font_size)
        styles.append(TextStyleInfo(
            text=" ".join(line["words"]),
            bbox=(x0, y0, x1 - x0, h),
            font_size_px=round(font_size, 1),
            weight=weight,
            line_height_px=float(h),
            suggestions=[dict(s) for s in _SANS_CANDIDATES],
            confidence=INFERIDO,
        ))
    if not styles:
        return [], "OCR ejecutado pero sin texto legible en el frame de referencia."
    return styles, None
