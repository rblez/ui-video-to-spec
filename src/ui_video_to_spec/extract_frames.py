"""Paso 1 — Extracción adaptativa de frames con ffmpeg.

Dos pasadas:

1. **Análisis**: se extrae una serie barata de frames de sondeo (4 fps) y se
   calcula la diferencia media entre consecutivos. Los tramos con diferencia
   alta son "ventanas de movimiento" (hay animación); el resto es estático.
   También se detectan cambios de escena con el filtro ``scene`` de ffmpeg.
2. **Extracción**: frames estáticos a ``fps_static`` (1 fps por defecto) en
   todo el video y frames densos a ``fps_motion`` (30–60) **solo** dentro de
   las ventanas de movimiento. Así una UI de 60 s no genera miles de PNG.

Todos los frames se guardan en PNG sin pérdida, con su timestamp real
(parseado de ``showinfo``) y un ``manifest.json`` que los describe.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .types import Config, Frame, VideoMeta

_PTS_RE = re.compile(r"pts_time:([0-9]*\.?[0-9]+)")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_video(video: str | Path) -> VideoMeta:
    """Devuelve metadatos del video usando ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video),
    ]
    res = _run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe falló sobre {video}: {res.stderr.strip()[:300]}")
    info = json.loads(res.stdout)
    vstream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vstream is None:
        raise RuntimeError(f"{video} no tiene pista de video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    num, den = (vstream.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(info["format"].get("duration") or vstream.get("duration") or 0.0)
    return VideoMeta(
        path=str(video),
        duration_s=duration,
        width=int(vstream["width"]),
        height=int(vstream["height"]),
        fps=fps,
        has_audio=has_audio,
    )


def _extract_segment(
    video: str,
    out_dir: Path,
    prefix: str,
    fps: float,
    start: float = 0.0,
    duration: float | None = None,
) -> list[Frame]:
    """Extrae frames a ``fps`` dentro de [start, start+duration) con timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{prefix}_%05d.png")
    cmd = ["ffmpeg", "-v", "info", "-ss", f"{start:.3f}", "-i", video]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps={fps},showinfo", "-fps_mode", "passthrough", pattern]
    res = _run(cmd)
    times = [float(t) for t in _PTS_RE.findall(res.stderr)]
    files = sorted(out_dir.glob(f"{prefix}_*.png"))
    frames = []
    for i, f in enumerate(files):
        ts = start + (times[i] if i < len(times) else i / fps)
        frames.append((f, ts))
    return frames


def _small_gray(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((160, 90)), dtype=np.float32)


def _frame_diff(pa: Path, pb: Path) -> float:
    """Diferencia absoluta media (0-255) entre dos frames, en miniatura."""
    return float(np.mean(np.abs(_small_gray(pa) - _small_gray(pb))))


def _changed_fraction(pa: Path, pb: Path) -> float:
    """Fracción de píxeles que cambian de forma apreciable (>18 niveles).

    Complementa a la diferencia media: un elemento pequeño que se anima
    (un botón en una pantalla grande) apenas mueve la media global, pero
    sí cambia casi todos sus píxeles. Esta señal lo detecta.
    """
    diff = np.abs(_small_gray(pa) - _small_gray(pb))
    return float(np.mean(diff > 18))


def _scene_times(video: str, duration: float) -> list[float]:
    """Tiempos de cambio de escena según el detector de ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "info", "-i", video,
        "-vf", "select='gt(scene,0.4)',showinfo", "-f", "null", "-",
    ]
    res = _run(cmd)
    times = [float(t) for t in _PTS_RE.findall(res.stderr)]
    return [t for t in times if 0.0 < t < duration]


def detect_motion_windows(
    video: str, work_dir: Path, meta: VideoMeta, threshold: float
) -> list[tuple[float, float]]:
    """Primera pasada: localiza ventanas [inicio, fin) con animación."""
    probe_dir = work_dir / "probe"
    probe_fps = 4.0
    raw = _extract_segment(video, probe_dir, "probe", probe_fps)
    if len(raw) < 2:
        return []
    hot: list[float] = []
    for (fa, ta), (fb, tb) in zip(raw, raw[1:]):
        if _frame_diff(fa, fb) > threshold or _changed_fraction(fa, fb) > 0.004:
            hot.extend([ta, tb])
    if not hot:
        return []
    step = 1.0 / probe_fps
    windows: list[tuple[float, float]] = []
    start = max(0.0, min(hot) - step - 0.25)
    end = min(meta.duration_s, max(hot) + step + 0.25)
    # Une marcas calientes separadas por menos de 1 s en una sola ventana.
    marks = sorted(set(round(t, 3) for t in hot))
    cur_start = marks[0]
    prev = marks[0]
    for t in marks[1:]:
        if t - prev > 1.0:
            windows.append((max(0.0, cur_start - step - 0.25),
                            min(meta.duration_s, prev + step + 0.25)))
            cur_start = t
        prev = t
    windows.append((max(0.0, cur_start - step - 0.25),
                    min(meta.duration_s, prev + step + 0.25)))
    return windows or [(start, end)]


def extract_frames(
    video: str | Path, work_dir: str | Path, cfg: Config
) -> tuple[VideoMeta, list[Frame], list[tuple[float, float]]]:
    """Ejecuta el paso 1 completo. Devuelve (meta, frames, ventanas)."""
    video = str(video)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video)

    windows = detect_motion_windows(video, work, meta, cfg.motion_diff_threshold)

    frames: list[Frame] = []
    # Frames estáticos en todo el video.
    for f, ts in _extract_segment(video, work / "static", "static", cfg.fps_static):
        frames.append(Frame(str(f), ts, "static"))
    # Frames densos solo en ventanas animadas.
    for wi, (w0, w1) in enumerate(windows):
        seg = _extract_segment(
            video, work / f"motion{wi}", f"motion{wi}", float(cfg.fps_motion),
            start=w0, duration=w1 - w0,
        )
        for f, ts in seg:
            frames.append(Frame(str(f), ts, "motion"))
    # Keyframes en cambios de escena (un frame exacto por corte).
    for si, t in enumerate(_scene_times(video, meta.duration_s)):
        out = work / "key" / f"key_{si:03d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video,
              "-frames:v", "1", str(out)])
        if out.is_file():
            frames.append(Frame(str(out), t, "key"))

    frames.sort(key=lambda fr: fr.timestamp)
    manifest = {
        "video": meta.path,
        "duration_s": meta.duration_s,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "motion_windows": [list(w) for w in windows],
        "frames": [
            {"file": fr.path, "timestamp": round(fr.timestamp, 3), "kind": fr.kind}
            for fr in frames
        ],
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta, frames, windows
