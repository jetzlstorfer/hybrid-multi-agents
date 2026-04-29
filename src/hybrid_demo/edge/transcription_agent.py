"""Edge transcription agent.

Wraps Foundry Local's Whisper model. This stage is fail-fast: if audio is
missing or transcription fails, the workflow errors instead of degrading.
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

    client = runtime.get_local_audio_client()
    if state.language_hint:
        client.settings.language = state.language_hint.split("-")[0]
    converted_path: Path | None = None
    try:
        result = client.transcribe(state.audio_uri)
    except Exception as exc:
        if not _looks_like_audio_format_error(exc):
            raise

        source = _to_local_path(state.audio_uri)
        converted_path = _convert_to_wav(source)
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
