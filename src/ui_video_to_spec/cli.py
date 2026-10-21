"""CLI de ui-video-to-spec.

Uso:
    ui-video-to-spec ruta/video.mp4 --out ./out --fps-motion 60

Orquesta los 8 pasos del pipeline y deja en ``--out``:
``design-tokens.json``, ``motion.json``, ``components/``, ``SKILL.md``,
``report.md`` y ``verify/``. Los frames intermedios viven en ``out/.work``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import audio, generate, interpret, measure_motion, measure_static
from . import typography, verify as verify_mod
from .extract_frames import extract_frames
from .types import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ui-video-to-spec",
        description="Convierte un video de UI en un design spec medido "
                    "(tokens, motion y componentes).",
    )
    p.add_argument("video", help="Video de la interfaz (mp4/mov/webm).")
    p.add_argument("--out", default="./out", help="Carpeta de salida.")
    p.add_argument("--fps-motion", type=int, default=None,
                   help="FPS en ventanas con animación (defecto: config).")
    p.add_argument("--fps-static", type=float, default=None,
                   help="FPS en zonas estáticas (defecto: config).")
    p.add_argument("--ssim-threshold", type=float, default=None,
                   help="Umbral SSIM del loop de verificación.")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Intentos máximos de verificación.")
    p.add_argument("--config", default=None, help="Ruta a config.yaml.")
    p.add_argument("--skip-verify", action="store_true",
                   help="Salta el paso 8 (verificación visual).")
    p.add_argument("--with-audio", action="store_true",
                   help="Transcribe el audio con Whisper (extra [audio]).")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.fps_motion is not None:
        cfg.fps_motion = args.fps_motion
    if args.fps_static is not None:
        cfg.fps_static = args.fps_static
    if args.ssim_threshold is not None:
        cfg.ssim_threshold = args.ssim_threshold
    if args.max_attempts is not None:
        cfg.max_attempts = args.max_attempts

    video = Path(args.video)
    if not video.is_file():
        print(f"Error: no existe el video {video}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".work"

    print(f"[1/8] Extrayendo frames de {video} …")
    meta, frames, windows = extract_frames(video, work, cfg)
    print(f"      {len(frames)} frames · ventanas de movimiento: "
          f"{[(round(a, 1), round(b, 1)) for a, b in windows]}")

    print("[2/8] Audio …")
    transcript, audio_note = ([], "Audio no solicitado (usa --with-audio).")
    if args.with_audio:
        transcript, audio_note = audio.transcribe(meta, cfg.language)
    if audio_note:
        print(f"      {audio_note}")

    print("[3/8] Midiendo estática (paleta, elementos, radios, sombras) …")
    m = measure_static.measure_static(frames, meta, cfg)
    m.transcript = transcript
    if audio_note:
        m.notes.append(audio_note)
    print(f"      {len(m.palette)} colores · {len(m.elements)} elementos · "
          f"espaciados {m.spacing_scale}")

    print("[4/8] Tipografía …")
    if m.reference_frame:
        styles, typo_note = typography.analyze_typography(m.reference_frame)
        m.text_styles = styles
        if typo_note:
            m.notes.append(typo_note)
        # Asocia el texto OCR al elemento que lo contiene (para componentes).
        for ts in m.text_styles:
            cx = ts.bbox[0] + ts.bbox[2] / 2
            cy = ts.bbox[1] + ts.bbox[3] / 2
            for el in m.elements:
                if (el.bbox[0] <= cx <= el.bbox[0] + el.bbox[2]
                        and el.bbox[1] <= cy <= el.bbox[1] + el.bbox[3]):
                    el.text = ts.text
                    break
    print(f"      {len(m.text_styles)} estilos de texto")

    print("[5/8] Midiendo movimiento …")
    motion_frames = [f for f in frames if f.kind == "motion"]
    bg_hex = m.palette[0].hex if m.palette else None
    m.animations = measure_motion.measure_motion(
        motion_frames, m.elements, bg_hex, cfg, meta.width)
    print(f"      {len(m.animations)} animaciones")

    print("[6/8] Interpretando componentes …")
    provider = interpret.interpret(
        m,
        key_frames=[f.path for f in frames
                    if f.kind in ("key", "static")][:4],
        video_path=str(video) if cfg.vision_native_video else None,
        cfg=cfg,
    )
    print(f"      proveedor: {provider.name}")

    print("[7/8] Generando entregables …")
    tokens = generate.write_json_outputs(m, out)
    components = generate.write_components(m, out)
    generate.write_skill(m, tokens, out)
    print(f"      {len(components)} componentes React")

    verify_summary = None
    if not args.skip_verify:
        print("[8/8] Verificando contra el video …")
        verify_summary = verify_mod.verify(m, out, cfg)
        print(f"      SSIM {verify_summary['ssim']} "
              f"(umbral {verify_summary['threshold']}, "
              f"método {verify_summary['method']})")
    else:
        print("[8/8] Verificación saltada (--skip-verify).")
    generate.write_report(m, out, verify_summary)

    print(f"\nListo. Salida en {out}/")
    print("  design-tokens.json · motion.json · components/ · "
          "SKILL.md · report.md" + (" · verify/" if verify_summary else ""))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
