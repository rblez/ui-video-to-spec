# ui-video-to-spec

Convierte un **video de una interfaz** (mp4/mov/webm) en un **design spec
medido**: tokens de diseño, animaciones con sus curvas reales, componentes
React + Tailwind y un skill autogenerado que describe el estilo para que
otros agentes lo reproduzcan casi idéntico.

> **Principio clave — medir con código, interpretar con visión.**
> Los números (hex, px, ms, cubic-bezier) salen de scripts que analizan
> los píxeles. El modelo con visión, si se usa, solo nombra componentes,
> jerarquía e intención. Nunca estima un valor a ojo.

## Salida

```
out/
├── design-tokens.json   # colores, espaciados, radios, sombras, tipografía
├── motion.json          # por animación: propiedad, duración, delay, easing, desde/hasta
├── components/          # React + Tailwind de los componentes detectados
├── SKILL.md             # skill autogenerado con el estilo capturado
├── report.md            # confianza por valor + lo que NO se puede deducir
└── verify/              # SSIM por intento y diffs visuales (paso 8)
```

Cada valor del reporte lleva su nivel de confianza:

- **medido** — calculado directamente de los píxeles/frames.
- **inferido** — derivado de una medición con una heurística (p. ej. el
  radio se estima del paseo diagonal en la esquina del contorno).
- **adivinado** — sugerencia sin medición directa (p. ej. la familia
  tipográfica parecida).

## Instalación

Requisitos del sistema: **ffmpeg y ffprobe** en el PATH, Python ≥ 3.10.

```bash
# Con uv (recomendado)
uvx ui-video-to-spec --help

# O desde el repo
git clone <este-repo> && cd ui-video-to-spec
python -m venv .venv && .venv/bin/pip install -e .

# Extras opcionales
.venv/bin/pip install -e ".[ocr]"     # OCR con pytesseract (+ binario tesseract)
.venv/bin/pip install -e ".[audio]"   # transcripción con Whisper
.venv/bin/pip install -e ".[verify]"  # render de verificación con Playwright
.venv/bin/pip install -e ".[vision]"  # requests para proveedores de visión
```

Dependencias core (mínimas): `numpy`, `opencv-python-headless`, `Pillow`,
`jinja2`. El pipeline MVP corre completo sin ningún extra.

## Uso

```bash
ui-video-to-spec grabacion.mp4 --out ./out --fps-motion 60

python -m ui_video_to_spec grabacion.webm --out ./out \
  --fps-static 1 --fps-motion 30 \
  --ssim-threshold 0.92 --max-attempts 3 \
  --config config.yaml --with-audio
```

### Ejemplo reproducible

```bash
# Genera un video UI sintético con verdad de terreno (fondo, card con
# radio y sombra, botón con fade+slide de easing conocido):
.venv/bin/python examples/make_example_video.py
# Corre el pipeline sobre él:
.venv/bin/ui-video-to-spec examples/ui-demo.mp4 --out examples/out
# Compara contra examples/expected.json: el botón debe salir con
# duración ≈600ms, delay ≈1000ms y easing ≈ cubic-bezier(0.22, 1, 0.36, 1).
```

## Cómo funciona el pipeline

| Paso | Módulo | Qué hace |
|---|---|---|
| 1 | `extract_frames.py` | Muestreo adaptativo en dos pasadas: sondeo a 4 fps para localizar ventanas de movimiento (+ detección de escenas) y extracción densa (30–60 fps) solo ahí; 1 fps en zonas estáticas. PNG sin pérdida + `manifest.json` con timestamps |
| 2 | `audio.py` | (Opcional) Whisper con timestamps alineado a los frames. Degrada con gracia si no hay audio o no está instalado |
| 3 | `measure_static.py` | Paleta exacta por clustering con **medianas** (robusto a compresión), bounding boxes por contornos, radios por análisis de esquinas, sombras por perfil de luminancia, espaciados frecuentes, jerarquía por contención |
| 4 | `typography.py` | OCR (pytesseract si está) + tamaño/peso/interlineado desde las cajas; sugiere 3 fuentes de Google Fonts con score (etiquetado *adivinado*) |
| 5 | `measure_motion.py` | Flujo óptico Farneback + diff entre frames → trayectorias por elemento: desplazamiento, escala y opacidad por frame; ajuste de `cubic-bezier` por least squares contra easings conocidos + búsqueda en rejilla; duración y delay |
| 6 | `interpret.py` | Nombra componentes y agrupa. Proveedor configurable por entorno: `none` (heurística geométrica local), `openai`, `anthropic`. Soporta modelos con video nativo (`vision_native_video`) |
| 7 | `generate.py` | Escribe tokens, motion, componentes `.tsx`, el `SKILL.md` del estilo y `report.md` desde `templates/` (Jinja2) |
| 8 | `verify.py` | Renderiza los componentes (Playwright o render sintético PIL) y compara con el original (SSIM + diff de color). Itera refinando rellenos hasta el umbral o N intentos; guarda diffs en `out/verify/` |

## Configuración

`config.yaml` (raíz del repo) es **plano a propósito** (`clave: valor`):
el loader propio (`types.load_config`) parsea ese subconjunto sin exigir
PyYAML. Claves: `fps_static`, `fps_motion`, `palette_size`,
`min_element_area`, `motion_diff_threshold`, `ssim_threshold`,
`max_attempts`, `vision_provider`, `vision_model`, `vision_native_video`,
`language`. Los flags del CLI tienen prioridad sobre el archivo.

Proveedor de visión por entorno:

```bash
export UI2SPEC_VISION_PROVIDER=anthropic   # none | openai | anthropic
export UI2SPEC_VISION_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=...
```

## Compresión de video

Los videos llegan comprimidos y el códec altera los colores. El pipeline:

- usa **medianas** de cluster en vez de píxeles sueltos para la paleta;
- detecta artefactos midiendo el *blockiness* (saltos de luminancia en
  bordes de bloques 8×8 vs. interior) y **avisa en `report.md`** cuando
  los colores pueden estar alterados.

## Limitaciones (honestas)

Un video solo muestra lo que se grabó. `report.md` incluye siempre la
sección **"Lo que NO se puede deducir de este video"**:

- estados hover/active/focus y navegación por teclado;
- estados de error, vacío, carga y éxito de formularios;
- dark mode u otros temas no grabados;
- comportamiento responsive en otros anchos;
- interacciones no ejecutadas durante la grabación;
- micro-animaciones bajo la resolución temporal efectiva.

Además: la detección de elementos es por color/contorno (una UI con
gradientes complejos o elementos del mismo color que el fondo se mide
peor), la tipografía sin OCR instalado no se mide, y la verificación con
render sintético no reproduce texto ni imágenes, así que su SSIM es una
cota orientativa, no una nota final.

## Estructura del repo

```
ui-video-to-spec/
├── README.md
├── pyproject.toml
├── config.yaml
├── skill/SKILL.md            # skill instalable para Claude Code u otro agente
├── src/ui_video_to_spec/     # un módulo por paso del pipeline + types/cli
├── templates/                # plantillas Jinja2 (SKILL, report, componente)
├── tests/                    # pytest: un test por paso + end-to-end
└── examples/                 # generador de video sintético + expected.json
```

## Tests

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Los tests generan un video pequeño con el mismo generador de
`examples/` y verifican las mediciones contra la verdad de terreno con
tolerancias (la compresión impide exigir igualdad exacta).
