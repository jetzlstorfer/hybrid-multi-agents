"""Edge transcription agent.

Dispatches to either faster-whisper (default; handles long audio natively via
internal 30-second chunking and VAD silence filtering) or Foundry Local's
Whisper model.  The active backend is selected by the ``TRANSCRIPTION_BACKEND``
env var (``faster-whisper`` | ``foundry``; default: ``faster-whisper``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from .. import runtime
from ..contracts import Transcript, TranscriptSegment, WorkflowState
from ..telemetry import traced_step


def _looks_like_audio_format_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cannot detect audio stream format" in msg


def _to_local_path(audio_uri: str) -> Path:
    if audio_uri.startswith("file://"):
        parsed = urlparse(audio_uri)
        return Path(unquote(parsed.path))
    return Path(audio_uri)


def _convert_to_wav(input_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Transcription failed because the audio format is not supported by the local "
            "decoder and `ffmpeg` is not installed. Install ffmpeg or upload a WAV/MP3 file."
        )

    fd, out_file = tempfile.mkstemp(prefix="hybrid_demo_", suffix=".wav")
    os.close(fd)
    out_path = Path(out_file)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        details = (proc.stderr or proc.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"Audio conversion via ffmpeg failed: {details}")

    return out_path


# ---------- faster-whisper backend ----------


def _get_faster_whisper_model():
    """Lazy-load and cache a faster-whisper WhisperModel."""
    import faster_whisper  # noqa: PLC0415

    model_name = os.environ.get("FASTER_WHISPER_MODEL", "small")
    device = os.environ.get("FASTER_WHISPER_DEVICE", "cpu")
    compute = os.environ.get("FASTER_WHISPER_COMPUTE", "int8")

    if not hasattr(_get_faster_whisper_model, "_model"):
        _get_faster_whisper_model._model = faster_whisper.WhisperModel(
            model_name,
            device=device,
            compute_type=compute,
        )
    return _get_faster_whisper_model._model


def _transcribe_faster_whisper(audio_path: Path, language: str | None) -> str:
    """Transcribe with faster-whisper, which handles long files via internal chunking.

    ``vad_filter=True`` strips leading/trailing silence so Whisper does not
    waste context window on dead air at the start of the recording.
    """
    model = _get_faster_whisper_model()
    lang = language.split("-")[0] if language else None
    segments, _info = model.transcribe(
        str(audio_path),
        language=lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )
    return " ".join(seg.text.strip() for seg in segments)


# ---------- Foundry Local backend ----------


@traced_step(
    name="edge.transcription",
    runtime_location="edge",
    input_classification="raw_audio",
    output_classification="raw_transcript",
    model_role="edge.transcription",
)
def transcribe(state: WorkflowState) -> Transcript:
    if state.audio_uri is None:
        raise ValueError("audio_uri is required")

    backend = os.environ.get("TRANSCRIPTION_BACKEND",
                             "faster-whisper").lower().strip()
    audio_path = _to_local_path(state.audio_uri)

    if backend == "faster-whisper":
        text = _transcribe_faster_whisper(audio_path, state.language_hint)
    else:
        # Foundry Local backend — processes only ~30 s by default; convert
        # format if the decoder cannot detect the stream.
        client = runtime.get_local_audio_client()
        if state.language_hint:
            client.settings.language = state.language_hint.split("-")[0]
        converted_path: Path | None = None
        try:
            result = client.transcribe(state.audio_uri)
        except Exception as exc:
            if not _looks_like_audio_format_error(exc):
                raise
            converted_path = _convert_to_wav(audio_path)
            result = client.transcribe(str(converted_path))
        finally:
            if converted_path is not None:
                try:
                    converted_path.unlink(missing_ok=True)
                except Exception:
                    pass
        text = result.text or ""

    # Foundry Local returns plain text. We wrap it as a single segment; speaker
    # diarisation is not in scope for the demo.
    return Transcript(
        workflow_id=state.workflow_id,
        transcript_id=f"tr_{state.workflow_id}",
        language=state.language_hint or "de-AT",
        segments=[TranscriptSegment(speaker="unknown", text=text)],
    )
