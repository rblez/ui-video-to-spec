---
name: ui-video-to-spec
description: "Convierte un video (mp4/mov/webm) de una interfaz en un design spec MEDIDO: design-tokens.json, motion.json, componentes React+Tailwind, SKILL.md del estilo y reporte con niveles de confianza. Úsalo cuando el usuario pase un video/grabación de una UI y quiera recrearla, clonarla, documentar su estilo o extraer sus tokens y animaciones."
---

# ui-video-to-spec — de video de UI a design spec medido

Principio: **medir con código, interpretar con visión**. Los números
(hex, px, ms, curvas de easing) los calculan scripts sobre los píxeles;
el modelo solo nombra componentes e intención. Nunca presentes como
medido un valor que no salga en los archivos de salida.

## Cuándo invocarlo

- El usuario adjunta o referencia un video de una app/web y pide
  recrearla, copiar su estilo, extraer sus colores/animaciones o
  documentar su diseño.
- Necesitas la base objetiva (tokens + motion) antes de construir una UI
  "como la del video".

## Requisitos

- `ffmpeg`/`ffprobe` en el PATH.
- El paquete instalado (incluye numpy, OpenCV headless, Pillow, Jinja2).
- Extras opcionales según el caso: `[ocr]` OCR, `[audio]` transcripción
  con Whisper, `[verify]` render con Playwright (sin él, la verificación
  usa un render sintético y lo anota).

## Cómo invocarlo

```bash
uvx ui-video-to-spec ruta/video.mp4 --out ./out --fps-motion 60
# o, con el repo clonado:
python -m ui_video_to_spec ruta/video.mp4 --out ./out \
  --fps-static 1 --fps-motion 30 --ssim-threshold 0.92 --max-attempts 3
```

Flags útiles: `--config config.yaml`, `--skip-verify`,
`--with-audio` (transcribe voz alineada a los frames).

## Cómo leer la salida (`./out`)

| Archivo | Qué contiene |
|---|---|
| `design-tokens.json` | Colores con rol (background/surface/primary/text) y confianza, escala de espaciado, radios, sombras, tipografía sugerida |
| `motion.json` | Por animación: elemento, propiedad, duración ms, delay ms, `cubic-bezier(...)` y valores desde/hasta |
| `components/*.tsx` | React + Tailwind con los valores medidos (andamiaje fiel, no diseño final) |
| `SKILL.md` | Skill autogenerado del estilo capturado, reutilizable por otros agentes |
| `report.md` | Confianza por valor (medido/inferido/adivinado), avisos y lo NO deducible del video |
| `verify/` | SSIM por intento y diffs visuales contra el frame original |

## Reglas al trabajar con el resultado

1. **Nunca inventes números.** Si un valor no está en los JSON o está
   marcado `adivinado`, dilo; no lo presentes como hecho.
2. Respeta la confianza: `medido` es fiable (salvo aviso de compresión),
   `inferido` es una estimación razonable, `adivinado` es una sugerencia.
3. Lo que el video no muestra **no existe** en el spec: hover, focus,
   errores, dark mode, responsive. Lee esa sección del `report.md` y
   defínela aparte con el usuario.
4. Si el report avisa de **artefactos de compresión**, los colores pueden
   desviarse unos puntos: la paleta ya usa medianas para mitigarlo.
5. Para recrear la UI, parte de `components/` y del `SKILL.md`
   generado; ajusta solo lo que el loop de verificación marque en
   `verify/` (SSIM y diff de color).

## Interpretación con IA (opcional)

Por defecto el paso de interpretación es heurístico (sin red). Para usar
un modelo con visión que nombre los componentes:

```bash
export UI2SPEC_VISION_PROVIDER=openai   # o anthropic
export UI2SPEC_VISION_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...               # o ANTHROPIC_API_KEY
```

El modelo solo agrupa y nombra; jamás modifica las mediciones.
