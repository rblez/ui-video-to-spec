"""Paso 7 — Generación de entregables en /out.

- ``design-tokens.json``: colores (con rol y confianza), escala de
  espaciado, radios, sombras y tipografía.
- ``motion.json``: animaciones medidas con duración, delay y cubic-bezier.
- ``components/*.tsx``: React + Tailwind con los valores medidos como
  estilos/arbitrary values. Son andamiaje fiel a la medición, no diseño final.
- ``SKILL.md``: skill autogenerado que describe el estilo para otros agentes.
- ``report.md``: confianza por valor + lo que NO se puede deducir del video.

Las plantillas viven en ``templates/`` y se renderizan con Jinja2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .types import Animation, Element, Measurements

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)),
                       autoescape=False, trim_blocks=True, lstrip_blocks=True)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "ui"


def _pascal(kind: str, el_id: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", " ", kind).title().replace(" ", "")
    return f"{base or 'Box'}{el_id.split('-')[-1]}"


# ---------------------------------------------------------------------------
# Tokens y motion (JSON)
# ---------------------------------------------------------------------------
def build_tokens(m: Measurements) -> dict:
    colors = {}
    for i, c in enumerate(m.palette):
        name = c.role or f"color-{i + 1}"
        colors[name] = {
            "value": c.hex,
            "type": "color",
            "proportion": round(c.proportion, 4),
            "confidence": c.confidence,
        }
    radii = {}
    for el in m.elements:
        if el.radius_px > 0:
            radii[f"radius-{el.radius_px}"] = {
                "value": el.radius_px, "type": "dimension",
                "confidence": "inferido",
            }
    shadows = {}
    seen_shadow = set()
    for el in m.elements:
        if el.shadow:
            css = el.shadow.to_css()
            if css not in seen_shadow:
                seen_shadow.add(css)
                shadows[f"shadow-{len(shadows) + 1}"] = {
                    "value": css, "type": "shadow",
                    "confidence": el.shadow.confidence,
                }
    typography = {}
    for i, t in enumerate(m.text_styles):
        typography[f"text-{i + 1}"] = {
            "fontSize": t.font_size_px,
            "fontWeight": t.weight,
            "lineHeight": t.line_height_px,
            "sample": t.text[:60],
            "suggestedFonts": t.suggestions,
            "confidence": t.confidence,
        }
    return {
        "$meta": {
            "generator": "ui-video-to-spec",
            "source": m.meta.path if m.meta else None,
            "resolution": [m.meta.width, m.meta.height] if m.meta else None,
        },
        "color": colors,
        "spacing": {
            "scale": {"value": m.spacing_scale, "type": "dimension[]",
                      "confidence": "inferido"},
        },
        "radius": radii,
        "shadow": shadows,
        "typography": typography,
    }


def build_motion(m: Measurements) -> dict:
    return {
        "$meta": {"generator": "ui-video-to-spec",
                  "source": m.meta.path if m.meta else None},
        "animations": [
            {
                "element": a.element_id,
                "property": a.property,
                "duration_ms": a.duration_ms,
                "delay_ms": a.delay_ms,
                "easing": list(a.easing),
                "easing_css": a.easing_css(),
                "easing_name": a.easing_name,
                "from": a.from_value,
                "to": a.to_value,
                "unit": a.unit,
                "confidence": a.confidence,
            }
            for a in m.animations
        ],
    }


# ---------------------------------------------------------------------------
# Componentes React + Tailwind
# ---------------------------------------------------------------------------
def _child_markup(el: Element, parent: Element) -> str:
    x = el.bbox[0] - parent.bbox[0]
    y = el.bbox[1] - parent.bbox[1]
    _, _, w, h = el.bbox
    style = [f"left: {x}", f"top: {y}", f"width: {w}", f"height: {h}"]
    if el.fill_hex:
        style.append(f'backgroundColor: "{el.fill_hex}"')
    if el.radius_px:
        style.append(f"borderRadius: {el.radius_px}")
    if el.shadow:
        style.append(f'boxShadow: "{el.shadow.to_css()}"')
    text = el.text.replace('"', "'") if el.text else ""
    return (
        f'      <div className="absolute" style={{{{ {", ".join(style)} }}}}'
        f'{">"}{text}</div>'
    )


def write_components(m: Measurements, out_dir: Path) -> list[str]:
    """Escribe Screen.tsx + un .tsx por componente con nombre. Devuelve rutas."""
    comp_dir = out_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)
    env = _env()
    template = env.get_template("component.tsx.j2")
    by_id = {e.id: e for e in m.elements}
    written: list[str] = []

    def render(name: str, root: Element, children: list[Element], comment: str):
        markup = "\n".join(_child_markup(c, root) for c in children)
        # El atributo style se construye aquí porque las llaves JSX {{ }}
        # chocan con la sintaxis de Jinja2 dentro de la plantilla.
        style_parts = [f"width: {root.bbox[2]}", f"height: {root.bbox[3]}"]
        if root.fill_hex:
            style_parts.append(f'backgroundColor: "{root.fill_hex}"')
        if root.radius_px:
            style_parts.append(f"borderRadius: {root.radius_px}")
        if root.shadow:
            style_parts.append(f'boxShadow: "{root.shadow.to_css()}"')
        style_attr = "style={{ " + ", ".join(style_parts) + " }}"
        content = template.render(
            name=name, comment=comment, style_attr=style_attr,
            children_markup=markup,
        )
        path = comp_dir / f"{name}.tsx"
        path.write_text(content, encoding="utf-8")
        written.append(str(path))

    screen = by_id.get("el-0")
    if screen is not None and m.meta:
        tops = [e for e in m.elements if e.id != "el-0"]
        render("Screen", screen, tops,
               f"Pantalla completa medida ({m.meta.width}x{m.meta.height}). "
               "Hijos posicionados con valores absolutos medidos.")
    for el in m.elements:
        if el.id == "el-0" or el.kind in ("box", "screen"):
            continue
        kids = [by_id[c] for c in el.children if c in by_id]
        render(_pascal(el.kind, el.id), el, kids,
               f"Componente '{el.kind}' medido en {el.bbox}.")
    return written


# ---------------------------------------------------------------------------
# SKILL.md y report.md
# ---------------------------------------------------------------------------
def write_skill(m: Measurements, tokens: dict, out_dir: Path) -> str:
    env = _env()
    video_name = Path(m.meta.path).stem if m.meta else "ui"
    colors = [
        {"name": name, "value": t["value"], "confidence": t["confidence"]}
        for name, t in tokens["color"].items()
    ]
    radii = [{"name": k, "value": v["value"]}
             for k, v in tokens["radius"].items()]
    shadows = [v["value"] for v in tokens["shadow"].values()]
    typography = [
        {"size": v["fontSize"], "weight": v["fontWeight"],
         "font": (v["suggestedFonts"][0]["name"] if v["suggestedFonts"] else "?"),
         "confidence": v["confidence"]}
        for v in tokens["typography"].values()
    ]
    animations = [
        {"property": a.property, "duration": a.duration_ms,
         "delay": a.delay_ms, "easing": a.easing_css(),
         "from": a.from_value, "to": a.to_value, "unit": a.unit}
        for a in m.animations
    ]
    content = env.get_template("skill.md.j2").render(
        slug=_slug(video_name), video_name=video_name,
        width=m.meta.width if m.meta else 0,
        height=m.meta.height if m.meta else 0,
        colors=colors, spacing_list=", ".join(str(s) for s in m.spacing_scale),
        radii=radii, shadows=shadows, typography=typography,
        animations=animations,
        bg=(m.palette[0].hex if m.palette else "#FFFFFF"),
    )
    path = out_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def write_report(m: Measurements, out_dir: Path,
                 verify_summary: dict | None = None) -> str:
    env = _env()
    rows = []
    for name, t in build_tokens(m)["color"].items():
        rows.append({"label": f"color.{name}", "value": t["value"],
                     "confidence": t["confidence"]})
    for el in m.elements:
        if el.id == "el-0":
            continue
        rows.append({
            "label": f"{el.kind} ({el.id}) bbox",
            "value": f"x={el.bbox[0]} y={el.bbox[1]} "
                     f"{el.bbox[2]}x{el.bbox[3]}px",
            "confidence": el.confidence,
        })
        if el.radius_px:
            rows.append({"label": f"{el.kind} ({el.id}) radio",
                         "value": f"{el.radius_px}px",
                         "confidence": "inferido"})
        if el.shadow:
            rows.append({"label": f"{el.kind} ({el.id}) sombra",
                         "value": el.shadow.to_css(),
                         "confidence": el.shadow.confidence})
    if m.spacing_scale:
        rows.append({"label": "escala de espaciado",
                     "value": ", ".join(str(s) for s in m.spacing_scale) + " px",
                     "confidence": "inferido"})
    for t in m.text_styles:
        font = t.suggestions[0]["name"] if t.suggestions else "?"
        rows.append({"label": f'tipografía "{t.text[:24]}"',
                     "value": f"{t.font_size_px}px peso {t.weight}, "
                              f"fuente sugerida {font}",
                     "confidence": t.confidence})
    animations = [
        {"element": a.element_id or "?", "property": a.property,
         "duration": a.duration_ms, "delay": a.delay_ms,
         "easing": a.easing_css(), "from": a.from_value, "to": a.to_value,
         "unit": a.unit, "confidence": a.confidence}
        for a in m.animations
    ]
    ref_time = 0.0
    content = env.get_template("report.md.j2").render(
        video_name=Path(m.meta.path).name if m.meta else "video",
        video_path=m.meta.path if m.meta else "",
        width=m.meta.width if m.meta else 0,
        height=m.meta.height if m.meta else 0,
        duration=round(m.meta.duration_s, 2) if m.meta else 0,
        ref_time=ref_time, rows=rows, animations=animations,
        warnings=m.warnings, notes=m.notes, transcript=m.transcript,
        verify=verify_summary,
    )
    path = out_dir / "report.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def write_json_outputs(m: Measurements, out_dir: Path) -> dict:
    tokens = build_tokens(m)
    (out_dir / "design-tokens.json").write_text(
        json.dumps(tokens, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "motion.json").write_text(
        json.dumps(build_motion(m), indent=2, ensure_ascii=False),
        encoding="utf-8")
    return tokens
