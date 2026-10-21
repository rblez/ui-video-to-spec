"""Paso 5 — Medición de movimiento entre frames densos.

- **Flujo óptico** (Farneback, OpenCV) + diferencia absoluta entre frames:
  la unión de ambas máscaras localiza qué se mueve en cada instante.
- **Tracking**: las regiones móviles se enlazan entre frames por IoU y forman
  trayectorias (centro, tamaño y color medio por frame).
- **Propiedades**: desplazamiento del centro → translateX/translateY;
  cambio de tamaño → scale; mezcla del color de la región hacia el fondo →
  opacity (ecuación de composición alfa).
- **Easing**: la curva de progreso normalizada p(t) se ajusta contra los
  cubic-bezier conocidos de CSS y luego se refina con búsqueda en rejilla
  (least squares). Duración = tramo activo; delay = timestamp de inicio.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .types import Animation, Config, Element, Frame

KNOWN_EASINGS: dict[str, tuple[float, float, float, float]] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease-in": (0.42, 0.0, 1.0, 1.0),
    "ease-out": (0.0, 0.0, 0.58, 1.0),
    "ease-in-out": (0.42, 0.0, 0.58, 1.0),
    "ease-out-quint": (0.22, 1.0, 0.36, 1.0),
    "ease-in-out-cubic": (0.65, 0.0, 0.35, 1.0),
}


def bezier_y(x: float, p: tuple[float, float, float, float]) -> float:
    """Evalúa y(x) de un cubic-bezier resolviendo el parámetro por bisección."""
    x1, y1, x2, y2 = p
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(20):
        u = (lo + hi) / 2
        xu = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u * u * x2 + u ** 3
        if xu < x:
            lo = u
        else:
            hi = u
    u = (lo + hi) / 2
    return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u * u * y2 + u ** 3


def _sse(params, ts: np.ndarray, ps: np.ndarray) -> float:
    pred = np.array([bezier_y(float(t), params) for t in ts])
    return float(np.sum((pred - ps) ** 2))


def fit_easing(ts: list[float], ps: list[float]) -> tuple[tuple, str, float]:
    """Ajusta un cubic-bezier a la curva de progreso (t, p en 0..1).

    Primero prueba los easings conocidos de CSS; después refina con una
    búsqueda en rejilla gruesa + local. Devuelve (params, nombre, error).
    """
    if len(ts) < 3:
        return KNOWN_EASINGS["ease"], "ease", 1.0
    t = np.asarray(ts, dtype=np.float64)
    p = np.asarray(ps, dtype=np.float64)
    # Normaliza y fuerza monotonía creciente (el progreso no retrocede).
    p = np.maximum.accumulate(np.clip(p, 0.0, 1.0))
    if p[-1] > 0:
        p = p / p[-1]
    best, best_name, best_err = None, "custom", float("inf")
    for name, params in KNOWN_EASINGS.items():
        err = _sse(params, t, p)
        if err < best_err:
            best, best_name, best_err = params, name, err
    # Rejilla gruesa (paso 0.2) y refinamiento local (paso 0.05).
    grid = [i / 5 for i in range(6)]
    cand = [(a, b, c, d) for a in grid for b in grid for c in grid for d in grid]
    for params in cand:
        err = _sse(params, t, p)
        if err < best_err:
            best, best_name, best_err = params, "custom", err
    bx1, by1, bx2, by2 = best
    fine = [i / 20 for i in range(-2, 3)]
    for dx1 in fine:
        for dy1 in fine:
            for dx2 in fine:
                for dy2 in fine:
                    params = (min(1.0, max(0.0, bx1 + dx1)), by1 + dy1,
                              min(1.0, max(0.0, bx2 + dx2)), by2 + dy2)
                    err = _sse(params, t, p)
                    if err < best_err:
                        best, best_name, best_err = params, "custom", err
    # Si un easing conocido queda casi igual, prefiere el nombre conocido.
    for name, params in KNOWN_EASINGS.items():
        if _sse(params, t, p) <= best_err * 1.15 + 1e-4:
            return params, name, best_err
    return tuple(round(v, 3) for v in best), best_name, best_err


def _load_gray(path: str, width: int = 640) -> np.ndarray:
    with Image.open(path) as im:
        w, h = im.size
        scale = width / w if w > width else 1.0
        im = im.convert("L")
        if scale != 1.0:
            im = im.resize((width, max(1, int(h * scale))))
        return np.asarray(im)


def _load_rgb_small(path: str, width: int = 640) -> np.ndarray:
    with Image.open(path) as im:
        w, h = im.size
        scale = width / w if w > width else 1.0
        if scale != 1.0:
            im = im.resize((width, max(1, int(h * scale))))
        return np.asarray(im.convert("RGB"))


def _iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _moving_boxes(prev_g, next_g, prev_c, scale: float) -> list[tuple]:
    """Cajas móviles entre dos frames (flujo óptico ∪ diferencia)."""
    flow = cv2.calcOpticalFlowFarneback(prev_g, next_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    diff = cv2.absdiff(prev_g, next_g)
    mask = ((mag > 0.5) | (diff > 10)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 120:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append(tuple(int(v / scale) for v in (x, y, w, h)))
    return boxes


def _match_element(bbox, elements: list[Element]) -> str | None:
    best, best_iou = None, 0.25
    for el in elements:
        if el.id == "el-0":
            continue
        v = _iou(bbox, el.bbox)
        if v > best_iou:
            best, best_iou = el.id, v
    return best


def measure_motion(
    motion_frames: list[Frame],
    elements: list[Element],
    bg_hex: str | None,
    cfg: Config,
    video_width: int,
) -> list[Animation]:
    """Detecta animaciones en las ventanas densas y ajusta sus curvas."""
    if len(motion_frames) < 3:
        return []
    frames = sorted(motion_frames, key=lambda f: f.timestamp)
    scale = 640 / video_width if video_width > 640 else 1.0
    grays = [_load_gray(f.path) for f in frames]
    colors = [_load_rgb_small(f.path) for f in frames]
    bg_rgb = None
    if bg_hex:
        bg_rgb = np.array([int(bg_hex[i:i + 2], 16) for i in (1, 3, 5)],
                          dtype=np.float64)

    # Tracking: trayectorias de cajas enlazadas por IoU entre frames.
    # Se comparan frames con un paso (stride) para que el cambio por paso
    # supere el ruido: a 60 fps una animación suave se mueve <1px/frame.
    if len(frames) >= 2:
        step = max(1, int(round(1.0 / max(1e-6, frames[1].timestamp
                                         - frames[0].timestamp) / 30)))
    else:
        step = 1
    indices = list(range(0, len(frames), step))
    tracks: list[list[dict]] = []
    for a, b in zip(indices, indices[1:]):
        boxes = _moving_boxes(grays[a], grays[b], colors[a], scale)
        for box in boxes:
            match = None
            for tr in tracks:
                if tr and tr[-1]["frame"] == a and _iou(box, tr[-1]["bbox"]) > 0.25:
                    match = tr
                    break
            sample = {"frame": b, "t": frames[b].timestamp, "bbox": box}
            if match is not None:
                match.append(sample)
            else:
                tracks.append([{"frame": a, "t": frames[a].timestamp,
                                "bbox": box}, sample])

    def region_color(frame_idx: int, box, mean: bool = False) -> np.ndarray | None:
        x, y, w, h = (int(v * scale) for v in box)
        img = colors[frame_idx]
        crop = img[max(0, y):y + max(1, h), max(0, x):x + max(1, w)]
        if not crop.size:
            return None
        flat = crop.reshape(-1, 3)
        fn = np.mean if mean else np.median
        return fn(flat, axis=0).astype(np.float64)

    def _overlap_fraction(a, b) -> float:
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        return (ix * iy) / max(1, bw * bh)

    animations: list[Animation] = []
    for track in tracks:
        if len(track) < 4:
            continue
        ts = [s["t"] for s in track]
        boxes = [s["bbox"] for s in track]
        duration_s = ts[-1] - ts[0]
        if duration_s < 0.05:
            continue
        centers = np.array([(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in boxes],
                           dtype=np.float64)
        widths = np.array([b[2] for b in boxes], dtype=np.float64)
        el_id = _match_element(boxes[-1], elements)
        delay_ms = int(round(ts[0] * 1000))

        # --- Serie alfa sobre la caja final FIJA + extensión del tramo ---
        # La cola de un ease-out es casi invisible al diff (progreso >97%),
        # así que la trayectoria detectada se corta antes del final real.
        # Muestreando el alfa de la región fija en frames posteriores se
        # recupera el final verdadero: cuando el alfa llega al 99% del final.
        alpha_series: list[tuple[float, float]] = []
        end_t = ts[-1]
        if bg_rgb is not None:
            fixed = boxes[-1]
            i0 = track[0]["frame"]
            base = region_color(max(0, i0 - step), fixed, mean=True)
            solid = region_color(len(frames) - 1, fixed, mean=True)
            if base is not None and solid is not None:
                axis_vec = solid - base
                denom = float(np.linalg.norm(axis_vec))
                if denom > 30:
                    # Cobertura: fracción de la caja fija que el elemento
                    # ocupa en cada instante (al deslizarse entra poco a
                    # poco). El alfa proyectado con la MEDIA de la región
                    # es cobertura×opacidad; se divide para aislar la
                    # opacidad real.
                    box_by_frame = {s["frame"]: s["bbox"] for s in track}
                    known = sorted(box_by_frame)
                    raw = []
                    for idx in range(i0, len(frames)):
                        c = region_color(idx, fixed, mean=True)
                        if c is None:
                            continue
                        proj = float(np.dot(c - base, axis_vec)
                                     / (denom ** 2))
                        cov = 1.0
                        if idx <= known[-1]:
                            near = max(k for k in known if k <= idx) \
                                if any(k <= idx for k in known) else known[0]
                            cov = max(0.2, _overlap_fraction(
                                box_by_frame[near], fixed))
                        a = float(np.clip(proj / cov, 0, 1))
                        raw.append((frames[idx].timestamp, a))
                    if raw:
                        a_final = raw[-1][1]
                        alpha_series = raw
                        if a_final > 0.1:
                            # Fin real: último instante bajo el 99% del
                            # alfa final (robusto a picos de ruido). Nota:
                            # la cola de un ease-out vive bajo el ruido de
                            # píxel, así que la duración medida es la del
                            # tramo *visible* de la animación.
                            below = [tval for tval, aval in raw
                                     if aval < 0.99 * a_final]
                            if below:
                                after = [tval for tval, _ in raw
                                         if tval > below[-1]]
                                if after:
                                    end_t = after[0]
        extended = end_t > ts[-1] + 1e-9
        if extended:
            duration_s = end_t - ts[0]
        duration_ms = int(round(duration_s * 1000))
        t_norm = [(t - ts[0]) / duration_s for t in ts]

        # --- translate (por eje, relativo a la posición final) ---
        for axis, prop in ((0, "translateX"), (1, "translateY")):
            series = centers[:, axis]
            total = series[-1] - series[0]
            if abs(total) >= 4:
                progress = list((series - series[0]) / total)
                tn = list(t_norm)
                if extended:  # la cola invisible llega al destino en t=1
                    tn.append(1.0)
                    progress.append(1.0)
                easing, name, _ = fit_easing(tn, progress)
                animations.append(Animation(
                    element_id=el_id, property=prop,
                    duration_ms=duration_ms, delay_ms=delay_ms,
                    easing=easing, easing_name=name,
                    from_value=round(float(-total), 1), to_value=0.0, unit="px",
                ))
        # --- scale ---
        if widths[0] > 0 and abs(widths[-1] / widths[0] - 1) >= 0.06:
            progress = list((widths - widths[0]) / (widths[-1] - widths[0]))
            tn = list(t_norm)
            if extended:
                tn.append(1.0)
                progress.append(1.0)
            easing, name, _ = fit_easing(tn, progress)
            animations.append(Animation(
                element_id=el_id, property="scale",
                duration_ms=duration_ms, delay_ms=delay_ms,
                easing=easing, easing_name=name,
                from_value=round(float(widths[0] / widths[-1]), 3), to_value=1.0,
                unit="",
            ))
        # --- opacity (alfa de la región fija, serie completa) ---
        if len(alpha_series) >= 4:
            seg = [(t, a) for t, a in alpha_series if t <= end_t + 1e-9]
            alphas = [a for _, a in seg]
            if max(alphas) - min(alphas) >= 0.15:
                tn = [(t - ts[0]) / duration_s for t, _ in seg]
                easing, name, _ = fit_easing(tn, alphas)
                animations.append(Animation(
                    element_id=el_id, property="opacity",
                    duration_ms=duration_ms, delay_ms=delay_ms,
                    easing=easing, easing_name=name,
                    from_value=round(alphas[0], 2),
                    to_value=round(alphas[-1], 2), unit="",
                ))
    # Consolida: misma propiedad+elemento+duración ≈ una sola animación.
    unique: list[Animation] = []
    for a in animations:
        if not any(u.element_id == a.element_id and u.property == a.property
                   and abs(u.duration_ms - a.duration_ms) < 60 for u in unique):
            unique.append(a)
    return unique
