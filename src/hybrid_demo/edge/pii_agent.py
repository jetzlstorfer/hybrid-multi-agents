"""Edge PII / sensitivity agent.

Entity extraction is SLM-only: a local Phi model returns structured JSON
which we validate against typed contracts. No deterministic regex/rule layer is
used in this stage.
"""

from __future__ import annotations

import json
from typing import Iterable

from ..contracts import (
    Entity,
    EntityType,
    SensitivityReport,
    Transcript,
    WorkflowState,
)
from ..policy import (
    CLOUD_FORBIDDEN_ENTITY_TYPES,
)
from ..telemetry import traced_step


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


def _parse_llm_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models include prose around the JSON block.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start: end + 1])


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
        repair_messages = [
            {"role": "system", "content": _JSON_REPAIR_PROMPT},
            {"role": "user", "content": raw},
        ]
        repaired = _complete_chat_raw(client, repair_messages)
        return _parse_llm_json(repaired)


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


def _slm_entities(text: str) -> list[Entity]:
    import os
    from .. import runtime

    client = runtime.get_local_chat_client()
    messages = [
        {"role": "system", "content": _PII_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    parsed = _complete_chat_json(client, messages)

    entities = _extract_entities(parsed, text)

    # Second pass: ask for missing entities only to improve recall.
    # Can be skipped via HYBRID_DEMO_SKIP_PII_AUDIT=1 to speed up the demo.
    skip_audit = os.environ.get(
        "HYBRID_DEMO_SKIP_PII_AUDIT", "").lower() in ("1", "true", "yes")
    if not skip_audit:
        audit_messages = [
            {"role": "system", "content": _PII_AUDIT_PROMPT},
            {
                "role": "user",
                "content": (
                    "TRANSKRIPT:\n"
                    + text
                    + "\n\nBISHERIGE_ENTITAETEN_JSON:\n"
                    + json.dumps(
                        {
                            "entities": [
                                {"type": ent.type, "value": ent.value}
                                for ent in entities
                            ]
                        },
                        ensure_ascii=False,
                    )
                ),
            },
        ]
        audit_parsed = _complete_chat_json(client, audit_messages)
        entities.extend(_extract_entities(audit_parsed, text))
    return _dedupe(entities)


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
