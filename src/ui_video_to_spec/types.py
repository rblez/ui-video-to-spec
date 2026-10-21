"""Tipos compartidos del pipeline y carga de configuración.

Principio del proyecto: **medir con código, interpretar con visión**.
Todo número que llega a los entregables (hex, px, ms, curvas) nace en un
script de medición; cada valor lleva un nivel de confianza:

- ``medido``    : calculado directamente de los píxeles/frames.
- ``inferido``  : derivado de una medición con una heurística (p. ej. radio
                  estimado desde la esquina de un contorno).
- ``adivinado`` : sugerencia sin medición directa (p. ej. nombre de fuente).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MEDIDO = "medido"
INFERIDO = "inferido"
ADIVINADO = "adivinado"


# ---------------------------------------------------------------------------
# Datos del pipeline
# ---------------------------------------------------------------------------
@dataclass
class VideoMeta:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass
class Frame:
    """Un frame extraído del video, con su timestamp en segundos."""

    path: str
    timestamp: float
    kind: str  # "static" | "motion" | "key"


@dataclass
class ColorInfo:
    hex: str
    rgb: tuple[int, int, int]
    proportion: float  # fracción de píxeles del frame de referencia
    role: str = ""  # background | surface | primary | text | ...
    confidence: str = MEDIDO


@dataclass
class ShadowInfo:
    offset_x: int
    offset_y: int
    blur: int
    color_hex: str
    opacity: float
    confidence: str = INFERIDO

    def to_css(self) -> str:
        r, g, b = (int(self.color_hex[i : i + 2], 16) for i in (1, 3, 5))
        return (
            f"{self.offset_x}px {self.offset_y}px {self.blur}px "
            f"rgba({r},{g},{b},{self.opacity:.2f})"
        )


@dataclass
class Element:
    id: str
    kind: str = "box"  # screen|card|button|navbar|input|modal|text|image|box
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h en px del video
    fill_hex: str | None = None
    radius_px: int = 0
    shadow: ShadowInfo | None = None
    confidence: str = MEDIDO
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    layer: int = 0
    text: str = ""


@dataclass
class TextStyleInfo:
    text: str
    bbox: tuple[int, int, int, int]
    font_size_px: float
    weight: int
    line_height_px: float
    suggestions: list[dict] = field(default_factory=list)  # [{name, score}]
    confidence: str = INFERIDO


@dataclass
class Animation:
    element_id: str | None
    property: str  # opacity | translateX | translateY | scale
    duration_ms: int
    delay_ms: int
    easing: tuple[float, float, float, float]  # cubic-bezier(x1, y1, x2, y2)
    easing_name: str
    from_value: float
    to_value: float
    unit: str  # "" para opacity/scale, "px" para translate
    confidence: str = MEDIDO

    def easing_css(self) -> str:
        x1, y1, x2, y2 = self.easing
        return f"cubic-bezier({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"


@dataclass
class Measurements:
    """Contenedor único que viaja por todo el pipeline."""

    meta: VideoMeta | None = None
    palette: list[ColorInfo] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)
    spacing_scale: list[int] = field(default_factory=list)
    text_styles: list[TextStyleInfo] = field(default_factory=list)
    animations: list[Animation] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    compression_detected: bool = False
    reference_frame: str | None = None  # frame usado para medir lo estático


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass
class Config:
    fps_static: float = 1.0
    fps_motion: int = 30
    palette_size: int = 8
    min_element_area: int = 900
    motion_diff_threshold: float = 3.0
    ssim_threshold: float = 0.92
    max_attempts: int = 3
    vision_provider: str = "none"
    vision_model: str = ""
    vision_native_video: bool = False
    language: str = "es"


def _parse_scalar(raw: str):
    """Convierte un escalar YAML simple a int/float/bool/str."""
    v = raw.strip().strip('"').strip("'")
    if v == "":
        return ""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_flat_yaml(text: str) -> dict:
    """Parser mínimo para YAML plano `clave: valor` (sin dependencias).

    Decisión documentada: en vez de exigir PyYAML, config.yaml se mantiene
    plano a propósito y este parser cubre ese subconjunto. Si el archivo
    tiene claves desconocidas se ignoran; si falta una clave, vale el
    defecto de :class:`Config`.
    """
    data: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        # Quita comentarios al final de línea solo si van precedidos de espacio.
        if " #" in value:
            value = value.split(" #", 1)[0]
        data[key.strip()] = _parse_scalar(value)
    return data


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Carga config.yaml (o defaults) y aplica overrides por entorno."""
    cfg = Config()
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates += [Path("config.yaml"), Path(__file__).resolve().parents[2] / "config.yaml"]
    for cand in candidates:
        if cand and cand.is_file():
            values = _parse_flat_yaml(cand.read_text(encoding="utf-8"))
            for key, val in values.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
            break
    # Overrides por variable de entorno (visión).
    if os.environ.get("UI2SPEC_VISION_PROVIDER"):
        cfg.vision_provider = os.environ["UI2SPEC_VISION_PROVIDER"]
    if os.environ.get("UI2SPEC_VISION_MODEL"):
        cfg.vision_model = os.environ["UI2SPEC_VISION_MODEL"]
    return cfg
