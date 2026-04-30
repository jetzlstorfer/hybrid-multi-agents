"""Edge PII / sensitivity agent.

Entity extraction is SLM-only: a local Phi model returns structured JSON
which we validate against typed contracts. No deterministic regex/rule layer is
used in this stage.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

from ..contracts import (
    Entity,
    EntityType,
    SensitivityReport,
    Transcript,
    WorkflowState,
)
from ..telemetry import traced_step

_log = logging.getLogger(__name__)


def _make_entity(etype: EntityType, value: str, placeholder: str) -> Entity:
    return Entity(
        type=etype,
        value=value,
        placeholder=placeholder,
    )


# ---------- SLM layer ----------

_PII_SYSTEM_PROMPT = """\
Du bist ein Datenschutz-Klassifikator für medizinische Gespräche.
Extrahiere Entitäten aus dem deutschen Transkript und gib AUSSCHLIESSLICH
gültiges JSON zurück, ohne Erklärungen.

Format:
{
    "entities": [
        {"type": "<TYPE>", "value": "<exact span from transcript>"}
    ]
}

Erlaubte TYPE-Werte:
PERSON_NAME, DATE_OF_BIRTH, AGE, ADDRESS, PHONE_NUMBER, EMAIL, INSURANCE_ID,
EMPLOYER, LOCATION, RELATIVE_NAME, MEDICAL_CONDITION, MEDICATION, SYMPTOM,
PROCEDURE, TIMESTAMP, FREE_TEXT_IDENTIFIER.

Pflichtregeln:
- Nutze nur exakte Textspannen aus dem Transkript (keine Paraphrasen).
- Erfinde nichts und normalisiere nichts.
- Wenn vorhanden, müssen direkte Identifikatoren extrahiert werden:
    PERSON_NAME, PHONE_NUMBER, EMAIL, INSURANCE_ID, ADDRESS, DATE_OF_BIRTH.
- Klinische Signale müssen extrahiert werden, falls vorhanden:
    MEDICAL_CONDITION, SYMPTOM, MEDICATION, PROCEDURE.
- Wenn nichts gefunden wurde: gib {"entities": []} zurück.
"""

_PII_AUDIT_PROMPT = """\
Prüfe die bisherige Entitätenliste gegen das Transkript.
Gib NUR fehlende Entitäten zurück (keine Duplikate, keine Erklärungen),
im exakt gleichen JSON-Format: {"entities": [{"type":"...","value":"..."}]}.
Nutze nur exakte Textspannen aus dem Transkript.
"""

_JSON_REPAIR_PROMPT = """\
Du reparierst ein fehlerhaftes JSON fuer Entitaeten.
Gib AUSSCHLIESSLICH valides JSON im Format
{"entities": [{"type": "...", "value": "..."}]}
zurueck. Keine Erklaerung, kein Markdown, keine Zusatztexte.
"""


def _placeholder_for(etype: EntityType) -> str:
    return {
        "PERSON_NAME": "[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]",
        "AGE": "[PATIENT_AGE]",
        "DATE_OF_BIRTH": "[DATE_OF_BIRTH]",
        "ADDRESS": "[ADDRESS]",
        "PHONE_NUMBER": "[PHONE_NUMBER]",
        "EMAIL": "[EMAIL]",
        "INSURANCE_ID": "[INSURANCE_ID]",
        "EMPLOYER": "[EMPLOYER]",
        "LOCATION": "[LOCATION]",
        "RELATIVE_NAME": "[RELATIVE]",
        "TIMESTAMP": "[TIMESTAMP]",
        "FREE_TEXT_IDENTIFIER": "[REDACTED]",
    }.get(etype, "")


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json / ``` wrappers that some models emit around JSON."""
    text = raw.strip()
    if text.startswith("```"):
        # Drop first line (```json or ```) and last ``` fence
        lines = text.splitlines()
        # Remove opening fence line
        lines = lines[1:]
        # Remove closing fence line if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_llm_json(raw: str) -> dict:
    """Parse the first valid JSON object from *raw*, tolerating surrounding prose.

    Uses ``raw_decode`` so that trailing text after the closing brace (e.g. a
    second JSON block, or a prose explanation) does not cause an 'Extra data'
    error.
    """
    stripped = _strip_markdown_fences(raw).strip()
    if not stripped:
        raise json.JSONDecodeError("Empty response", raw, 0)

    decoder = json.JSONDecoder()
    # Fast path: entire string is valid JSON.
    try:
        return decoder.decode(stripped)
    except json.JSONDecodeError:
        pass

    # Slow path: find the first '{' and decode from there, ignoring trailing
    # content (handles prose-wrapped or double-emitted JSON blocks).
    start = stripped.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    try:
        obj, _ = decoder.raw_decode(stripped, start)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    raise json.JSONDecodeError("Could not parse JSON object", raw, start)


def _complete_chat_raw(client: object, messages: list[dict[str, str]]) -> str:
    if hasattr(client, "complete_chat"):
        # Foundry Local ChatClient currently accepts only messages/tools.
        response = client.complete_chat(messages=messages)
    else:
        response = client.complete(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    return response.choices[0].message.content or "{}"


def _complete_chat_json(client: object, messages: list[dict[str, str]]) -> dict:
    raw = _complete_chat_raw(client, messages)
    try:
        return _parse_llm_json(raw)
    except json.JSONDecodeError:
        # Reasoning models can emit near-JSON with minor syntax issues.
        # Attempt a single repair pass; if that also fails, return an empty
        # result so a single bad chunk never kills the whole pipeline.
        try:
            repair_messages = [
                {"role": "system", "content": _JSON_REPAIR_PROMPT},
                {"role": "user", "content": raw},
            ]
            repaired = _complete_chat_raw(client, repair_messages)
            return _parse_llm_json(repaired)
        except json.JSONDecodeError:
            import logging
            logging.getLogger(__name__).warning(
                "PII agent: could not parse SLM JSON after repair; "
                "treating chunk as having no entities. Raw response: %r", raw[:200]
            )
            return {"entities": []}


def _extract_entities(payload: dict, source_text: str) -> list[Entity]:
    out: list[Entity] = []
    source_folded = source_text.casefold()
    for item in payload.get("entities", []):
        etype = item.get("type")
        value = item.get("value")
        if not etype or not value:
            continue
        if etype not in _ALL_TYPES:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        # Guard against hallucinations: keep only spans present in transcript.
        if value_str.casefold() not in source_folded:
            continue
        out.append(
            _make_entity(
                etype,
                value_str,
                _placeholder_for(etype) or "[REDACTED]",
            )
        )
    return out


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of at most *max_chars* on sentence boundaries.

    Splitting on '. ' keeps each chunk self-contained enough for PII
    extraction without losing spans that straddle sentence borders.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        # Find the last sentence boundary before the limit.
        cut = remaining.rfind(". ", 0, max_chars)
        if cut == -1:
            # No sentence boundary — hard-cut at last space to avoid splitting a word.
            cut = remaining.rfind(" ", 0, max_chars)
        if cut == -1:
            cut = max_chars
        else:
            cut += 1  # include the period / space in this chunk
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _split_chunk_for_retry(chunk: str) -> tuple[str, str]:
    """Split a chunk into two parts near the midpoint for retry."""
    if len(chunk) < 2:
        return chunk, ""

    mid = len(chunk) // 2
    cut = chunk.rfind(". ", 0, mid)
    if cut == -1:
        cut = chunk.rfind(" ", 0, mid)
    if cut == -1:
        cut = mid
    else:
        cut += 1

    left = chunk[:cut].strip()
    right = chunk[cut:].strip()
    if not left or not right:
        return chunk[:mid].strip(), chunk[mid:].strip()
    return left, right


def _is_operation_cancelled_error(exc: Exception) -> bool:
    return "operation was cancelled" in str(exc).lower()


def _slm_entities_for_chunk(
    client: object,
    chunk: str,
    *,
    min_chunk_chars: int,
) -> list[Entity]:
    messages = [
        {"role": "system", "content": _PII_SYSTEM_PROMPT},
        {"role": "user", "content": chunk},
    ]
    try:
        parsed = _complete_chat_json(client, messages)
        return _extract_entities(parsed, chunk)
    except Exception as exc:
        # Foundry Local can cancel over-large requests. Retry with smaller
        # chunks before giving up.
        if _is_operation_cancelled_error(exc) and len(chunk) > min_chunk_chars:
            left, right = _split_chunk_for_retry(chunk)
            _log.warning(
                "PII agent: chunk cancelled by runtime; retrying with smaller chunks "
                "(len=%d -> %d + %d)",
                len(chunk),
                len(left),
                len(right),
            )
            out: list[Entity] = []
            if left:
                out.extend(
                    _slm_entities_for_chunk(
                        client,
                        left,
                        min_chunk_chars=min_chunk_chars,
                    )
                )
            if right:
                out.extend(
                    _slm_entities_for_chunk(
                        client,
                        right,
                        min_chunk_chars=min_chunk_chars,
                    )
                )
            return out

        _log.warning(
            "PII agent: chunk failed with %s; treating as no entities",
            type(exc).__name__,
        )
        return []


def _slm_entities(text: str) -> list[Entity]:
    import os
    from .. import runtime

    # phi-4-mini's context window is large but the Foundry Local SDK cancels
    # requests when the prompt exceeds its internal inference limit.  We chunk
    # the transcript to stay well within the model's safe range (~2 000 chars
    # ≈ ~500 tokens of system prompt + ~500 tokens of user text).
    chunk_size = int(os.environ.get("HYBRID_DEMO_PII_CHUNK_CHARS", "1800"))
    min_chunk_chars = int(os.environ.get(
        "HYBRID_DEMO_PII_MIN_CHUNK_CHARS", "600"))
    if min_chunk_chars < 200:
        min_chunk_chars = 200
    if chunk_size < min_chunk_chars:
        chunk_size = min_chunk_chars
    chunks = _chunk_text(text, chunk_size)

    client = runtime.get_local_chat_client()
    all_entities: list[Entity] = []
    for chunk in chunks:
        all_entities.extend(
            _slm_entities_for_chunk(
                client,
                chunk,
                min_chunk_chars=min_chunk_chars,
            )
        )

    return _dedupe(all_entities)


_ALL_TYPES: frozenset[str] = frozenset({
    "PERSON_NAME", "DATE_OF_BIRTH", "AGE", "ADDRESS", "PHONE_NUMBER", "EMAIL",
    "INSURANCE_ID", "EMPLOYER", "LOCATION", "RELATIVE_NAME", "MEDICAL_CONDITION",
    "MEDICATION", "SYMPTOM", "PROCEDURE", "TIMESTAMP", "FREE_TEXT_IDENTIFIER",
})


# ---------- Merge + dedupe ----------


def _dedupe(entities: Iterable[Entity]) -> list[Entity]:
    seen: dict[tuple[str, str], Entity] = {}
    for ent in entities:
        key = (ent.type, ent.value.strip().lower())
        if key not in seen:
            seen[key] = ent
    return list(seen.values())


# ---------- Executor ----------


@traced_step(
    name="edge.pii",
    runtime_location="edge",
    input_classification="raw_transcript",
    output_classification="sensitivity_report",
    model_role="edge.slm",
)
def detect_pii(state: WorkflowState) -> SensitivityReport:
    transcript: Transcript = state.transcript  # type: ignore[assignment]
    full_text = "\n".join(seg.text for seg in transcript.segments)
    merged = _dedupe(_slm_entities(full_text))
    return SensitivityReport(workflow_id=state.workflow_id, entities=merged)
