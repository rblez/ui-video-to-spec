"""Paso 2 (opcional) — Transcripción con Whisper, alineada a los frames.

Degradación elegante: si el video no tiene audio, si Whisper no está
instalado (extra ``[audio]``) o si la transcripción falla, devuelve una
lista vacía y una nota que termina en ``report.md``. Nunca rompe el pipeline.
"""

from __future__ import annotations

from .types import VideoMeta


def transcribe(
    meta: VideoMeta, language: str = "es"
) -> tuple[list[dict], str | None]:
    """Devuelve (segmentos, nota). Segmento: {start, end, text} en segundos."""
    if not meta.has_audio:
        return [], "El video no tiene pista de audio; transcripción omitida."
    try:
        import whisper  # type: ignore  # openai-whisper (extra [audio])
    except ImportError:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            return [], (
                "Whisper no está instalado (extra [audio]); "
                "transcripción omitida."
            )
        # Ruta faster-whisper.
        try:
            model = WhisperModel("tiny")
            segments, _info = model.transcribe(meta.path, language=language)
            return (
                [
                    {"start": round(s.start, 2), "end": round(s.end, 2),
                     "text": s.text.strip()}
                    for s in segments
                ],
                None,
            )
        except Exception as exc:  # pragma: no cover - depende del entorno
            return [], f"La transcripción con faster-whisper falló: {exc}"
    # Ruta openai-whisper.
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(meta.path, language=language)
        return (
            [
                {"start": round(s["start"], 2), "end": round(s["end"], 2),
                 "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
            None,
        )
    except Exception as exc:  # pragma: no cover - depende del entorno
        return [], f"La transcripción con Whisper falló: {exc}"
