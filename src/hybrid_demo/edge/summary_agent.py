"""Edge summary agent.

The local SLM produces a structured ``HandoverPackage`` from the redacted
transcript. Strict JSON output. The prompt forbids diagnosis and the
re-introduction of identifiers.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts import (
    HandoverPackage,
    PatientContext,
    RedactedTranscript,
    Symptom,
    WorkflowState,
)
from ..telemetry import traced_step


_SUMMARY_SYSTEM_PROMPT = """\
Du erhältst ein redigiertes Arzt-Patienten-Transkript ohne direkte Identifikatoren.
Erzeuge AUSSCHLIESSLICH gültiges JSON nach dem unten beschriebenen Schema.
Keine Diagnose, keine Spekulation, keine Erfindung von Werten.
Verwende keine direkten Identifikatoren (Namen, Geburtsdaten, Adressen).

Schema:
{
  "patient_context": {
    "age_bucket": "string | null",
    "sex_or_gender": "string | null",
    "region_type": "string | null"
  },
  "chief_complaint": "string | null",
  "symptoms": [
    {
      "name": "string",
      "duration": "string | null",
      "severity": "string | null",
      "associated_symptoms": ["string"]
    }
  ],
  "known_medications": ["string"],
  "known_conditions": ["string"],
  "negative_findings": ["string"],
  "uncertainties": ["string"]
}
"""


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _as_str_or_none(item)
        if text is not None:
            out.append(text)
    return out


def _normalise_symptoms(value: Any) -> list[Symptom]:
    if not isinstance(value, list):
        return []

    out: list[Symptom] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue

        name = _as_str_or_none(raw_item.get("name"))
        if name is None:
            # Ignore malformed symptom items instead of failing the full stage.
            continue

        out.append(
            Symptom(
                name=name,
                duration=_as_str_or_none(raw_item.get("duration")),
                severity=_as_str_or_none(raw_item.get("severity")),
                associated_symptoms=_as_list_of_str(
                    raw_item.get("associated_symptoms")),
            )
        )
    return out


def _normalise_str_list(value: Any) -> list[str]:
    return _as_list_of_str(value)


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks on sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind(". ", 0, max_chars)
        if cut == -1:
            cut = remaining.rfind(" ", 0, max_chars)
        if cut == -1:
            cut = max_chars
        else:
            cut += 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _merge_parsed(results: list[dict]) -> dict:
    """Merge multiple per-chunk summary dicts into one coherent result.

    The first non-null value wins for scalar fields; list fields are unioned.
    """
    merged: dict = {}
    for result in results:
        # Scalar: patient_context
        if not merged.get("patient_context"):
            ctx = result.get("patient_context") or {}
            if any(v for v in ctx.values()):
                merged["patient_context"] = ctx

        # Scalar: chief_complaint — first non-null wins
        if not merged.get("chief_complaint"):
            merged["chief_complaint"] = result.get("chief_complaint")

        # Lists: accumulate
        for key in ("symptoms", "known_medications", "known_conditions",
                    "negative_findings", "uncertainties"):
            existing = merged.get(key, []) or []
            incoming = result.get(key) or []
            merged[key] = existing + incoming

    return merged


async def _slm_summary(redacted_text: str) -> dict:
    import os
    from .. import runtime

    # Use a dedicated chunk size for the summary stage so it can be tuned
    # independently of the PII stage.  The Foundry Local native library has a
    # ~120 s per-request inference timeout; keeping chunks small ensures each
    # call completes well within that window on CPU-only hardware.
    chunk_size = int(os.environ.get("HYBRID_DEMO_SUMMARY_CHUNK_CHARS", "600"))
    chunks = _chunk_text(redacted_text, chunk_size)

    agent = runtime.get_local_agent(
        "edge.slm",
        name="EdgeSummaryAgent",
        instructions=_SUMMARY_SYSTEM_PROMPT,
        response_format={"type": "json_object"},
    )
    results: list[dict] = []
    for chunk in chunks:
        results.append(await _slm_summary_chunk(agent, chunk))

    if len(results) == 1:
        return results[0]
    return _merge_parsed(results)


async def _slm_summary_chunk(agent: object, redacted_text: str, *, _attempt: int = 1) -> dict:
    """Call the local SLM via the Agent Framework, retrying once on transient cancellation.

    The Foundry Local native library enforces a ~120 s per-request timeout.
    When the model is slow (cold start, CPU-only), it may cancel mid-flight
    with "Operation was cancelled".  A single retry catches one-off blips
    without masking genuine failures.
    """
    import logging
    _chunk_log = logging.getLogger(__name__)

    try:
        response = await agent.run(redacted_text)
    except Exception as exc:
        # Retry once on transient "Operation was cancelled" from the native
        # Foundry Local library (its internal ~120 s inference timeout).
        cancelled = "cancelled" in str(
            exc).lower() or "canceled" in str(exc).lower()
        if cancelled and _attempt == 1:
            _chunk_log.warning(
                "Foundry Local cancelled chunk (attempt %d) — retrying: %s", _attempt, exc
            )
            return await _slm_summary_chunk(agent, redacted_text, _attempt=2)
        raise

    raw = _strip_markdown_fences(response.text or "{}").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some SLM outputs append extra prose after a valid JSON object.
        # Decode only the first JSON object from the response text.
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = raw.find("{")
        if start == -1:
            raise
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                "Expected JSON object", raw, start) from exc
        if isinstance(parsed, dict):
            return parsed
        raise json.JSONDecodeError("Expected JSON object", raw, start)


@traced_step(
    name="edge.summary",
    runtime_location="edge",
    input_classification="redacted_transcript",
    output_classification="cloud_handover_package",
    model_role="edge.slm",
)
async def summarise(state: WorkflowState) -> HandoverPackage:
    redacted: RedactedTranscript = state.redacted  # type: ignore[assignment]
    text = "\n".join(seg.text for seg in redacted.redacted_segments)
    parsed = await _slm_summary(text)

    return HandoverPackage(
        workflow_id=state.workflow_id,
        patient_context=PatientContext(
            **(parsed.get("patient_context") or {})),
        chief_complaint=_as_str_or_none(parsed.get("chief_complaint")),
        symptoms=_normalise_symptoms(parsed.get("symptoms", [])),
        known_medications=_normalise_str_list(
            parsed.get("known_medications", [])),
        known_conditions=_normalise_str_list(
            parsed.get("known_conditions", [])),
        negative_findings=_normalise_str_list(
            parsed.get("negative_findings", [])),
        uncertainties=_normalise_str_list(parsed.get("uncertainties", [])),
        forbidden_fields_removed=True,
    )
