"""Paso 6 — Interpretación con visión: nombrar, no medir.

El proveedor de visión (o la heurística local) solo decide **qué es** cada
elemento (botón, card, input…) y cómo se agrupan los componentes. Los
números ya están medidos; este paso jamás los toca.

Proveedores (``UI2SPEC_VISION_PROVIDER`` / config ``vision_provider``):

- ``none`` (defecto): heurística geométrica local, sin red ni API key.
  Clasifica por tamaño, posición, radio, saturación y contención.
- ``openai`` / ``anthropic``: llamada HTTP simple (urllib) con los frames
  clave en base64 + el JSON de medidas. Si falta la API key o algo falla,
  se cae a la heurística y se anota en el report.

Modelos con video nativo: si ``vision_native_video`` es true, el proveedor
recibe la ruta del video y el paso 6 no necesita frames (la extracción de
los pasos 1/3/5 sigue siendo necesaria para medir).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

from .types import ColorInfo, Config, Element, Measurements


# ---------------------------------------------------------------------------
# Heurística geométrica (proveedor "none" y fallback de los demás)
# ---------------------------------------------------------------------------
def _saturation(rgb: tuple[int, int, int]) -> float:
    mx, mn = max(rgb), min(rgb)
    return 0.0 if mx == 0 else (mx - mn) / mx


def classify_element(el: Element, meta_w: int, meta_h: int) -> str:
    """Nombre de componente a partir solo de geometría medida."""
    x, y, w, h = el.bbox
    area_ratio = (w * h) / max(1, meta_w * meta_h)
    if el.id == "el-0" or area_ratio > 0.85:
        return "screen"
    if el.children and area_ratio > 0.08:
        return "card"
    if y < meta_h * 0.14 and w > meta_w * 0.6:
        return "navbar"
    if 24 <= h <= 72 and w >= 64 and el.radius_px >= 4:
        return "button"
    if h >= 30 and h <= 64 and w > meta_w * 0.25:
        return "input"
    if area_ratio > 0.25 and el.children:
        return "modal"
    if h <= 28 and w >= 30:
        return "text"
    if w >= 40 and h >= 40:
        return "image"
    return "box"


def assign_color_roles(palette: list[ColorInfo]) -> None:
    """Asigna roles semánticos a la paleta según proporción/saturación."""
    if not palette:
        return
    palette[0].role = "background"
    rest = palette[1:]
    # Superficie: el color claro más presente tras el fondo.
    light = [c for c in rest if sum(c.rgb) / 3 > 200]
    if light:
        max(light, key=lambda c: c.proportion).role = "surface"
    # Primario: el más saturado con presencia mínima.
    saturated = [c for c in rest if c.proportion > 0.004]
    if saturated:
        primary = max(saturated, key=lambda c: _saturation(c.rgb))
        if _saturation(primary.rgb) > 0.25 and not primary.role:
            primary.role = "primary"
    # Texto: el más oscuro.
    dark = min(rest, key=lambda c: sum(c.rgb)) if rest else None
    if dark is not None and sum(dark.rgb) / 3 < 90 and not dark.role:
        dark.role = "text"
    for i, c in enumerate(palette):
        if not c.role:
            c.role = f"color-{i + 1}"


class VisionProvider:
    """Interfaz mínima de un proveedor de visión."""

    name = "base"
    supports_native_video = False

    def name_components(
        self, m: Measurements, key_frames: list[str], video_path: str | None
    ) -> dict[str, str]:
        """Devuelve {element_id: nombre_componente}. Vacío = sin opinión."""
        return {}


class HeuristicProvider(VisionProvider):
    name = "none"

    def name_components(self, m, key_frames, video_path):
        if m.meta is None:
            return {}
        return {
            el.id: classify_element(el, m.meta.width, m.meta.height)
            for el in m.elements
        }


class _HttpVisionProvider(VisionProvider):
    """Base para proveedores HTTP con imágenes (OpenAI/Anthropic)."""

    api_key_env = ""
    endpoint = ""

    def _frames_b64(self, key_frames: list[str]) -> list[str]:
        out = []
        for p in key_frames[:4]:
            try:
                out.append(base64.b64encode(Path(p).read_bytes()).decode())
            except OSError:
                continue
        return out

    def _prompt(self, m: Measurements) -> str:
        summary = [
            {"id": el.id, "bbox": el.bbox, "fill": el.fill_hex,
             "radius_px": el.radius_px, "children": el.children}
            for el in m.elements
        ]
        return (
            "Eres un clasificador de componentes UI. Recibes frames de una "
            "interfaz y sus elementos YA MEDIDOS (no inventes números). "
            "Responde SOLO un JSON {\"element_id\": \"componente\"} con "
            "valores entre: screen, navbar, card, button, input, modal, "
            "text, image, box.\nElementos: " + json.dumps(summary)
        )

    def name_components(self, m, key_frames, video_path):
        raise NotImplementedError


class OpenAIProvider(_HttpVisionProvider):
    name = "openai"
    api_key_env = "OPENAI_API_KEY"
    endpoint = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = "gpt-4o-mini", native_video: bool = False):
        self.model = model
        self.supports_native_video = native_video

    def name_components(self, m, key_frames, video_path):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError("OPENAI_API_KEY no definida")
        content: list[dict] = [{"type": "text", "text": self._prompt(m)}]
        for b64 in self._frames_b64(key_frames):
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64}"}})
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        return json.loads(text)


class AnthropicProvider(_HttpVisionProvider):
    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-sonnet-4-5", native_video: bool = False):
        self.model = model
        self.supports_native_video = native_video

    def name_components(self, m, key_frames, video_path):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY no definida")
        content: list[dict] = [{"type": "text", "text": self._prompt(m)}]
        for b64 in self._frames_b64(key_frames):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}})
        body = json.dumps({
            "model": self.model, "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}],
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=body,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []))
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1]) if start >= 0 else {}


def get_provider(cfg: Config) -> VisionProvider:
    name = (cfg.vision_provider or "none").lower()
    if name == "openai":
        return OpenAIProvider(cfg.vision_model or "gpt-4o-mini",
                              cfg.vision_native_video)
    if name == "anthropic":
        return AnthropicProvider(cfg.vision_model or "claude-sonnet-4-5",
                                 cfg.vision_native_video)
    return HeuristicProvider()


def interpret(
    m: Measurements, key_frames: list[str], video_path: str | None, cfg: Config
) -> VisionProvider:
    """Aplica nombres de componente y roles de color sobre las medidas."""
    assign_color_roles(m.palette)
    provider = get_provider(cfg)
    try:
        names = provider.name_components(m, key_frames, video_path)
        if names:
            for el in m.elements:
                if el.id in names:
                    el.kind = str(names[el.id])
            m.notes.append(
                f"Interpretación de componentes con el proveedor "
                f"'{provider.name}' (solo nombres; números intactos).")
            return provider
    except Exception as exc:
        m.notes.append(
            f"Proveedor de visión '{provider.name}' falló ({exc}); "
            "se usa la heurística geométrica local.")
    fallback = HeuristicProvider()
    for el in m.elements:
        if el.kind == "box" or provider.name != "none":
            el.kind = fallback.name_components(m, key_frames, video_path).get(
                el.id, el.kind)
    m.notes.append("Interpretación de componentes con heurística geométrica "
                   "local (sin IA): nombres por tamaño/posición/radio.")
    return provider
