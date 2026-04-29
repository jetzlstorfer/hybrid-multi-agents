"""Edge redaction agent.

Uses the local SLM to apply all replacements and transformations from a
structured replacement plan. Direct identifier values are still stored in the
local vault for edge-only rehydration.
"""

from __future__ import annotations

import json

from .. import policy, vault
from ..contracts import (
    Entity,
    RedactedTranscript,
    SensitivityReport,
    Transcript,
    TranscriptSegment,
    WorkflowState,
)
from ..telemetry import traced_step


_REDACTION_SYSTEM_PROMPT = """\
Du bist ein Redaktions-Agent fuer medizinische Gespraeche.
Du erhaeltst:
1) Ein Transkriptsegment.
2) Eine Liste von Regeln als {"type": "...", "source": "...", "action": "...", "replacement": "..."}.

Aufgabe:
- Fuehre ALLE Ersetzungen im Text konsistent durch.
- Bei PERSON_NAME-Ersetzungen ersetze auch reine Nachnamen-Anreden im Kontext
    (z. B. "Herr Mayer", "Frau Schmidt") mit demselben replacement.
- Transformationen werden von dir ausgefuehrt (nicht deterministisch im Code):
  - action=generalize_age: alter zu Altersgruppe (z. B. "42" -> "age bucket 40-49")
  - action=generalize_region: Ort/Adresse zu grobem Regionstyp (z. B. "urban area"/"rural area")
  - action=generalize_employer: Arbeitgeber zu nicht-identifizierender Kategorie.
- Belasse alles andere unveraendert.
- Gib AUSSCHLIESSLICH valides JSON im Format {"redacted_text": "..."} zurueck.
"""

_LLM_TRANSFORM_ACTIONS: dict[str, str] = {
    "AGE": "generalize_age",
    "ADDRESS": "generalize_region",
    "DATE_OF_BIRTH": "generalize_age",
    "LOCATION": "generalize_region",
    "EMPLOYER": "generalize_employer",
}


def _split_name(full_name: str) -> tuple[str | None, str | None]:
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def _replacement_for_entity(ent: Entity) -> str:
    if ent.type == "PERSON_NAME":
        first, last = _split_name(ent.value)
        if first and last:
            return "[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]"
        if first:
            return "[PATIENT_FIRST_NAME]"
        return ent.placeholder or "[REDACTED]"

    return ent.placeholder or "[REDACTED]"


def _parse_redaction_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _redact_segment_with_slm(text: str, entities: list[Entity]) -> str:
    """Delegate replacement behavior to the local SLM (fail-fast)."""
    from .. import runtime

    replacements: list[dict[str, str]] = []
    for ent in sorted(entities, key=lambda e: len(e.value), reverse=True):
        if not ent.value:
            continue
        if ent.type in policy.CLOUD_ALLOWED_CLINICAL:
            continue
        replacements.append(
            {
                "type": ent.type,
                "source": ent.value,
                "action": _LLM_TRANSFORM_ACTIONS.get(ent.type, "replace"),
                "replacement": _replacement_for_entity(ent),
            }
        )

    if not replacements:
        return text

    payload = {
        "text": text,
        "replacements": replacements,
    }

    client = runtime.get_local_chat_client()
    messages = [
        {"role": "system", "content": _REDACTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    if hasattr(client, "complete_chat"):
        response = client.complete_chat(messages=messages)
    else:
        response = client.complete(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

    raw = response.choices[0].message.content or "{}"
    parsed = _parse_redaction_json(raw)
    out = str(parsed.get("redacted_text", "")).strip()
    if not out:
        raise ValueError("Redaction SLM returned empty redacted_text")

    return out


def _build_vault_mapping(entities: list[Entity]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for ent in entities:
        if ent.type not in policy.CLOUD_FORBIDDEN_ENTITY_TYPES:
            continue
        if ent.type == "PERSON_NAME":
            first, last = _split_name(ent.value)
            if first:
                mapping["[PATIENT_FIRST_NAME]"] = first
            if last:
                mapping["[PATIENT_LAST_NAME]"] = last
            continue
        if ent.placeholder:
            mapping[ent.placeholder] = ent.value
    return mapping


@traced_step(
    name="edge.redaction",
    runtime_location="edge",
    input_classification="raw_transcript",
    output_classification="redacted_transcript",
)
def redact(state: WorkflowState) -> RedactedTranscript:
    transcript: Transcript = state.transcript  # type: ignore[assignment]
    sensitivity: SensitivityReport = state.sensitivity  # type: ignore[assignment]

    # Build the placeholder vault: only direct identifiers go in. Generalised
    # quasi-identifiers are not reversible, so they don't need a vault entry.
    vault_map = _build_vault_mapping(sensitivity.entities)
    if vault_map:
        vault.store(state.workflow_id, vault_map)

    redacted_segments = [
        TranscriptSegment(
            speaker=seg.speaker,
            text=_redact_segment_with_slm(seg.text, sensitivity.entities),
        )
        for seg in transcript.segments
    ]

    return RedactedTranscript(
        workflow_id=state.workflow_id,
        redacted_transcript_id=f"rtr_{state.workflow_id}",
        redacted_segments=redacted_segments,
    )
