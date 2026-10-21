"""Paso 3 — Medición estática sobre frames sin animación.

Todo sale de los píxeles, nada de intuición:

- **Paleta exacta**: clustering por color con *medianas*. El video viene
  comprimido (H.264/VP9), así que un píxel suelto miente; la mediana de cada
  cluster es robusta frente a ese ruido.
- **Artefactos de compresión**: se mide el "blockiness" — si los saltos de
  luminancia en los bordes de bloques 8×8 son mucho mayores que dentro de
  los bloques, los colores están alterados y se avisa en el report.
- **Elementos**: máscaras por color dominante → contornos (OpenCV) →
  bounding boxes. El fondo es el color con mayor proporción.
- **Radios de borde**: paseo diagonal desde cada esquina del bbox hasta el
  primer píxel del elemento; para un rect redondeado, esa distancia ``d``
  cumple ``r ≈ d / (√2 − 1)``.
- **Sombras**: perfil de luminancia en la franja bajo el elemento; una
  caída suave y localizada = sombra (offset = fila más oscura, blur ≈
  ancho de la caída).
- **Espaciados**: huecos frecuentes entre cajas vecinas y márgenes.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .types import (
    INFERIDO,
    MEDIDO,
    ColorInfo,
    Element,
    Measurements,
    ShadowInfo,
    VideoMeta,
)


def rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c)) for c in rgb))


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# ---------------------------------------------------------------------------
# Paleta y compresión
# ---------------------------------------------------------------------------
def extract_palette(img: np.ndarray, n_colors: int = 8) -> list[ColorInfo]:
    """Clustering por bins de color con centro = mediana de los miembros."""
    small = np.asarray(
        Image.fromarray(img).resize((min(320, img.shape[1]),
                                     max(1, int(img.shape[0] * min(320, img.shape[1]) / img.shape[1]))))
    ).reshape(-1, 3).astype(np.float32)
    if len(small) == 0:
        return []
    bins = (small // 16).astype(np.int32)
    keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    counts = Counter(keys.tolist())
    total = len(small)
    palette: list[ColorInfo] = []
    for key, count in counts.most_common():
        if len(palette) >= n_colors:
            break
        members = small[keys == key]
        center = np.median(members, axis=0)  # mediana: robusta a compresión
        palette.append(ColorInfo(
            hex=rgb_to_hex(center),
            rgb=tuple(int(round(c)) for c in center),
            proportion=count / total,
        ))
    # Fusiona centros casi idénticos que el binning haya partido.
    merged: list[ColorInfo] = []
    for c in palette:
        dup = next((m for m in merged
                    if np.linalg.norm(np.array(m.rgb) - np.array(c.rgb)) < 14), None)
        if dup:
            dup.proportion += c.proportion
        else:
            merged.append(c)
    return merged


def compression_blockiness(img: np.ndarray) -> float:
    """Ratio salto-en-borde-de-bloque / salto-interior. >~1.35 = artefactos."""
    gray = np.asarray(
        Image.fromarray(img).convert("L"), dtype=np.float32)
    dx = np.abs(np.diff(gray, axis=1))
    if dx.shape[1] < 16:
        return 1.0
    cols = np.arange(dx.shape[1])
    boundary = dx[:, (cols % 8) == 7]
    interior = dx[:, (cols % 8) != 7]
    if interior.mean() < 1e-6:
        return 1.0
    return float(boundary.mean() / interior.mean())


# ---------------------------------------------------------------------------
# Geometría de elementos
# ---------------------------------------------------------------------------
def _estimate_radius(mask: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Radio de esquina desde el paseo diagonal (ver docstring del módulo)."""
    s = max(4, min(w, h) // 3)
    corners = [(x, y, 1, 1), (x + w - 1, y, -1, 1),
               (x, y + h - 1, 1, -1), (x + w - 1, y + h - 1, -1, -1)]
    radii = []
    H, W = mask.shape
    for cx, cy, dx, dy in corners:
        d = 0.0
        for step in range(1, s):
            px, py = cx + dx * step, cy + dy * step
            if not (0 <= px < W and 0 <= py < H):
                break
            if mask[py, px]:
                d = step * math.sqrt(2)
                break
        if d > 0:
            radii.append(d / (math.sqrt(2) - 1))
    if not radii:
        return 0
    r = int(round(float(np.median(radii))))
    return max(0, min(r, min(w, h) // 2))


def _detect_shadow(img: np.ndarray, bbox, bg_rgb) -> ShadowInfo | None:
    """Busca una caída suave de luminancia justo bajo el elemento."""
    x, y, w, h = bbox
    H, W = img.shape[:2]
    y2 = y + h
    if y2 + 4 >= H or w < 12:
        return None
    strip = img[y2:min(H, y2 + 40), x + 4:max(x + 5, x + w - 4)].astype(np.float32)
    if strip.size == 0:
        return None
    bg_lum = float(np.dot(bg_rgb, [0.299, 0.587, 0.114]))
    row_lum = strip @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    row_med = np.median(row_lum, axis=1)
    dip = bg_lum - row_med  # positivo = más oscuro que el fondo
    idx = int(np.argmax(dip))
    if dip[idx] < 5 or dip[idx] > 90:
        return None
    dark_rows = np.where(dip > 3)[0]
    if len(dark_rows) < 2 or dark_rows[0] > 6:
        return None
    blur = int(max(2, (dark_rows[-1] - dark_rows[0]) // 1))
    rows = strip[max(0, idx - 1):idx + 2].reshape(-1, 3)
    color = np.median(rows, axis=0)
    return ShadowInfo(
        offset_x=0,
        offset_y=int(idx + 1),
        blur=blur,
        color_hex=rgb_to_hex(color),
        opacity=round(min(0.6, float(dip[idx]) / 255.0 * 2.2), 2),
    )


def find_elements(
    img: np.ndarray, palette: list[ColorInfo], min_area: int
) -> list[Element]:
    """Segmenta elementos por color dominante y mide su geometría."""
    H, W = img.shape[:2]
    frame_area = H * W
    bg_rgb = np.array(palette[0].rgb if palette else (255, 255, 255), dtype=np.float32)
    elements: list[Element] = [Element(
        id="el-0", kind="screen", bbox=(0, 0, W, H),
        fill_hex=palette[0].hex if palette else None, layer=0,
    )]
    kernel = np.ones((5, 5), np.uint8)
    eid = 1
    # Umbral adaptativo por color: la mitad de la distancia al color de
    # paleta más cercano. Un umbral fijo fundía colores próximos (una card
    # blanca sobre fondo #EEF1F6 dista ~24: con 26 la máscara lo cubría
    # todo y el elemento se descartaba por gigante).
    centers = [np.array(c.rgb, dtype=np.float32) for c in palette]
    for ci, color in enumerate(palette[1:], start=1):
        target = centers[ci]
        others = [np.linalg.norm(target - o)
                  for j, o in enumerate(centers) if j != ci]
        nearest = min(others) if others else 30.0
        thresh = float(np.clip(0.45 * nearest, 8.0, 30.0))
        dist = np.linalg.norm(img.astype(np.float32) - target, axis=2)
        mask = (dist < thresh).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > 0.92 * frame_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 8 or h < 8:
                continue
            radius = _estimate_radius(mask, x, y, w, h)
            shadow = _detect_shadow(img, (x, y, w, h), bg_rgb)
            elements.append(Element(
                id=f"el-{eid}", kind="box", bbox=(int(x), int(y), int(w), int(h)),
                fill_hex=color.hex, radius_px=radius, shadow=shadow,
                confidence=MEDIDO,
            ))
            eid += 1
    # Dedupe: cajas casi idénticas (IoU > 0.9) se quedan con la mayor.
    def iou(a, b) -> float:
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    unique: list[Element] = []
    for el in sorted(elements, key=lambda e: -(e.bbox[2] * e.bbox[3])):
        if all(iou(el.bbox, u.bbox) <= 0.9 for u in unique if u.id != "el-0"):
            unique.append(el)
    # Segunda pasada: duplicados por contención casi total y áreas parecidas
    # (la sombra difuminada de una card se segmenta como una caja gemela
    # ligeramente mayor). Se conserva el relleno con más presencia en la
    # paleta: el color real gana al gris de la sombra.
    proportion = {c.hex: c.proportion for c in palette}

    def containment(inner, outer) -> float:
        ix = max(0, min(inner[0] + inner[2], outer[0] + outer[2])
                 - max(inner[0], outer[0]))
        iy = max(0, min(inner[1] + inner[3], outer[1] + outer[3])
                 - max(inner[1], outer[1]))
        return (ix * iy) / max(1, inner[2] * inner[3])

    dropped: set[int] = set()
    for i, a in enumerate(unique):
        if i in dropped or a.id == "el-0":
            continue
        for j, b in enumerate(unique):
            if i in dropped:
                break
            if i == j or j in dropped or b.id == "el-0":
                continue
            area_a, area_b = a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3]
            ratio = max(area_a, area_b) / max(1, min(area_a, area_b))
            smaller, bigger = (a, b) if area_a <= area_b else (b, a)
            if ratio < 1.3 and containment(smaller.bbox, bigger.bbox) > 0.95:
                pa = proportion.get(a.fill_hex or "", 0.0)
                pb = proportion.get(b.fill_hex or "", 0.0)
                dropped.add(j if pa >= pb else i)
    elements = sorted([e for k, e in enumerate(unique) if k not in dropped],
                      key=lambda e: (e.bbox[1], e.bbox[0]))
    for i, el in enumerate(elements):
        el.id = f"el-{i}"
    # Jerarquía: padre = menor caja que contiene el centro; capa = profundidad.
    for el in elements:
        if el.id == "el-0":
            continue
        cx, cy = el.bbox[0] + el.bbox[2] / 2, el.bbox[1] + el.bbox[3] / 2
        candidates = [
            o for o in elements
            if o is not el
            and o.bbox[0] <= cx <= o.bbox[0] + o.bbox[2]
            and o.bbox[1] <= cy <= o.bbox[1] + o.bbox[3]
            and (o.bbox[2] * o.bbox[3]) > (el.bbox[2] * el.bbox[3])
        ]
        if candidates:
            parent = min(candidates, key=lambda o: o.bbox[2] * o.bbox[3])
            el.parent_id = parent.id
            parent.children.append(el.id)
    by_id = {e.id: e for e in elements}
    for el in elements:
        depth, node = 0, el
        while node.parent_id and node.parent_id in by_id:
            depth += 1
            node = by_id[node.parent_id]
        el.layer = depth
    return elements


def measure_spacings(elements: list[Element], width: int, height: int) -> list[int]:
    """Huecos frecuentes entre cajas y márgenes al borde del frame."""
    gaps: Counter[int] = Counter()
    boxes = [e.bbox for e in elements if e.id != "el-0"]
    for (x, y, w, h) in boxes:
        for g in (x, y, width - (x + w), height - (y + h)):
            if 0 < g <= 200:
                gaps[int(round(g / 2.0)) * 2] += 1
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            gx = max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))
            gy = max(b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
            for g in (gx, gy):
                if 0 < g <= 200:
                    gaps[int(round(g / 2.0)) * 2] += 1
    scale = [g for g, c in gaps.most_common(10) if c >= 1]
    return sorted(set(scale))[:8]


# ---------------------------------------------------------------------------
# Entrada del paso 3
# ---------------------------------------------------------------------------
def measure_static(
    frames: list, meta: VideoMeta, cfg, palette_size: int | None = None
) -> Measurements:
    """Mide paleta/elementos/espaciados sobre el frame estático de referencia.

    El frame de referencia es el último estático: la UI ya "asentada".
    """
    m = Measurements(meta=meta)
    statics = [f for f in frames if f.kind in ("static", "key")]
    if not statics:
        m.notes.append("Sin frames estáticos; medición estática omitida.")
        return m
    ref = statics[-1]
    m.reference_frame = ref.path
    img = load_rgb(ref.path)
    m.palette = extract_palette(img, palette_size or cfg.palette_size)
    ratio = compression_blockiness(img)
    if ratio > 1.35:
        m.compression_detected = True
        m.warnings.append(
            f"Artefactos de compresión detectados (blockiness {ratio:.2f} > 1.35): "
            "los colores pueden estar alterados por el códec; la paleta usa "
            "medianas de cluster para mitigarlo."
        )
    m.elements = find_elements(img, m.palette, cfg.min_element_area)
    m.spacing_scale = measure_spacings(m.elements, meta.width, meta.height)
    m.notes.append(
        f"Medición estática sobre el frame de referencia "
        f"(t={ref.timestamp:.2f}s, {ref.kind})."
    )
    return m
