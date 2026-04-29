"""Edge redaction agent.

Uses the local SLM to apply all replacements and transformations from a
structured replacement plan. Direct identifier values are still stored in the
local vault for edge-only rehydration.
"""

from __future__ import annotations

import json
import logging
import re

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

_log = logging.getLogger(__name__)


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

_GERMAN_HONORIFICS = ("Herr", "Herrn", "Frau")


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


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_redaction_json(raw: str) -> dict:
    stripped = _strip_markdown_fences(raw).strip()
    if not stripped:
        raise json.JSONDecodeError("Empty response", raw, 0)
    decoder = json.JSONDecoder()
    try:
        return decoder.decode(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    obj, _ = decoder.raw_decode(stripped, start)
    if isinstance(obj, dict):
        return obj
    raise json.JSONDecodeError("Could not parse JSON object", raw, start)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of at most *max_chars* on sentence boundaries."""
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


def _redact_segment_with_slm(text: str, entities: list[Entity]) -> str:
    """Delegate replacement behavior to the local SLM (fail-fast).

    Long segments are split into chunks that fit within the model's context
    window before being sent to the SLM; results are rejoined afterwards.
    """
    import os
    chunk_size = int(os.environ.get("HYBRID_DEMO_PII_CHUNK_CHARS", "3000"))
    chunks = _chunk_text(text, chunk_size)
    return " ".join(
        _redact_chunk_with_slm(chunk, entities) for chunk in chunks
    )


def _redact_chunk_with_slm(text: str, entities: list[Entity]) -> str:
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
    try:
        parsed = _parse_redaction_json(raw)
        out = str(parsed.get("redacted_text", "")).strip()
    except json.JSONDecodeError:
        _log.warning(
            "Redaction agent: malformed SLM JSON; falling back to deterministic replacement. "
            "Raw response prefix: %r",
            raw[:200],
        )
        out = _redact_chunk_deterministic(text, entities)
    if not out:
        out = _redact_chunk_deterministic(text, entities)

    # Enforce direct-identifier redaction even when the SLM misses specific
    # mentions (e.g. surname-only references like "Herr Gerster").
    return _enforce_direct_identifier_redaction(out, entities)


def _enforce_direct_identifier_redaction(text: str, entities: list[Entity]) -> str:
    out = _redact_chunk_deterministic(text, entities)
    return _enforce_person_name_mentions(out, entities)


def _enforce_person_name_mentions(text: str, entities: list[Entity]) -> str:
    out = text
    for ent in entities:
        if ent.type != "PERSON_NAME" or not ent.value:
            continue

        first, last = _split_name(ent.value)

        # Redact explicit title+surname references while preserving the title
        # token and casing from the source text.
        if last:
            title_pattern = re.compile(
                rf"\b({'|'.join(_GERMAN_HONORIFICS)})\s+{re.escape(last)}\b",
                flags=re.IGNORECASE,
            )
            out = title_pattern.sub(r"\1 [PATIENT_LAST_NAME]", out)

        # Redact standalone surname/first-name mentions.
        if last:
            out = re.sub(
                rf"\b{re.escape(last)}\b",
                "[PATIENT_LAST_NAME]",
                out,
                flags=re.IGNORECASE,
            )
        if first:
            first_replacement = "[PATIENT_FIRST_NAME]"
            if not last:
                first_replacement = _replacement_for_entity(ent)
            out = re.sub(
                rf"\b{re.escape(first)}\b",
                first_replacement,
                out,
                flags=re.IGNORECASE,
            )

    return out


def _redact_chunk_deterministic(text: str, entities: list[Entity]) -> str:
    """Fallback redaction if SLM output is malformed.

    Applies exact-string replacements for cloud-forbidden entities only.
    This is intentionally conservative and keeps the workflow progressing.
    """
    out = text
    for ent in sorted(entities, key=lambda e: len(e.value), reverse=True):
        if not ent.value:
            continue
        if ent.type in policy.CLOUD_ALLOWED_CLINICAL:
            continue
        replacement = _replacement_for_entity(ent)
        out = out.replace(ent.value, replacement)
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
    # type: ignore[assignment]
    sensitivity: SensitivityReport = state.sensitivity

    # Build the placeholder vault: only direct identifiers go in. Generalised
    # quasi-identifiers are not reversible, so they don't need a vault entry.
    vault_map = _build_vault_mapping(sensitivity.entities)
    if vault_map:
        vault.store(state.workflow_id, vault_map)

    redacted_segments = [
        TranscriptSegment(
            speaker=seg.speaker,
            text=_enforce_direct_identifier_redaction(
                _redact_segment_with_slm(seg.text, sensitivity.entities),
                sensitivity.entities,
            ),
        )
        for seg in transcript.segments
    ]

    return RedactedTranscript(
        workflow_id=state.workflow_id,
        redacted_transcript_id=f"rtr_{state.workflow_id}",
        redacted_segments=redacted_segments,
    )
