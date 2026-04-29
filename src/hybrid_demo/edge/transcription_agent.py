"""Edge transcription agent.

Wraps Foundry Local's Whisper model. This stage is fail-fast: if audio is
missing or transcription fails, the workflow errors instead of degrading.
"""

from __future__ import annotations

from .. import runtime
from ..contracts import Transcript, TranscriptSegment, WorkflowState
from ..telemetry import traced_step


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
    result = client.transcribe(state.audio_uri)
    text = result.text or ""

    # Foundry Local returns plain text. We wrap it as a single segment; speaker
    # diarisation is not in scope for the demo.
    return Transcript(
        workflow_id=state.workflow_id,
        transcript_id=f"tr_{state.workflow_id}",
        language=state.language_hint or "de-AT",
        segments=[TranscriptSegment(speaker="unknown", text=text)],
    )
