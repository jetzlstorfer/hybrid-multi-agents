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


def _slm_summary(redacted_text: str) -> dict:
    from .. import runtime

    client = runtime.get_local_chat_client()
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": redacted_text},
    ]
    if hasattr(client, "complete_chat"):
        response = client.complete_chat(messages=messages)
    else:
        response = client.complete(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )

    raw = (response.choices[0].message.content or "{}").strip()
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
def summarise(state: WorkflowState) -> HandoverPackage:
    redacted: RedactedTranscript = state.redacted  # type: ignore[assignment]
    text = "\n".join(seg.text for seg in redacted.redacted_segments)
    parsed = _slm_summary(text)

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
