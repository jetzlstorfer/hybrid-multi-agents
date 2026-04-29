"""Determinism + transformation tests for the redaction agent."""

from __future__ import annotations

from hybrid_demo.contracts import (
    Entity,
    SensitivityReport,
    Transcript,
    TranscriptSegment,
    WorkflowState,
)
from hybrid_demo.edge.redaction_agent import redact


def _state(text: str) -> WorkflowState:
    s = WorkflowState(workflow_id="wf_test")
    s.transcript = Transcript(
        workflow_id="wf_test",
        transcript_id="tr_test",
        language="de-AT",
        segments=[TranscriptSegment(speaker="patient", text=text)],
    )
    s.sensitivity = SensitivityReport(
        workflow_id="wf_test",
        entities=[
            Entity(
                type="PERSON_NAME",
                value="Anna Müller",
                placeholder="[PATIENT_NAME]",
            ),
            Entity(
                type="AGE",
                value="42",
                placeholder="[PATIENT_AGE]",
            ),
            Entity(
                type="ADDRESS",
                value="Wien",
                placeholder="[ADDRESS]",
            ),
            Entity(
                type="SYMPTOM",
                value="Atemnot",
                placeholder="[REDACTED]",
            ),
        ],
    )
    return s


def _state_with_entities(text: str, entities: list[Entity]) -> WorkflowState:
    s = WorkflowState(workflow_id="wf_test")
    s.transcript = Transcript(
        workflow_id="wf_test",
        transcript_id="tr_test",
        language="de-AT",
        segments=[TranscriptSegment(speaker="patient", text=text)],
    )
    s.sensitivity = SensitivityReport(workflow_id="wf_test", entities=entities)
    return s


def test_redaction_is_deterministic(monkeypatch):
    text = "Ich bin Anna Müller, 42, aus Wien."

    def fake_slm(segment: str, _entities: list[Entity]) -> str:
        return segment.replace("Anna Müller", "[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]")

    import hybrid_demo.edge.redaction_agent as redaction_agent

    monkeypatch.setattr(redaction_agent, "_redact_segment_with_slm", fake_slm)

    a = redact(_state(text))
    b = redact(_state(text))
    assert a.redacted_segments[0].text == b.redacted_segments[0].text


def test_quasi_identifiers_are_generalised(monkeypatch):
    text = "Ich bin Anna Müller, 42, aus Wien und habe Atemnot."

    def fake_slm(segment: str, _entities: list[Entity]) -> str:
        out = segment
        out = out.replace("Anna Müller", "[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]")
        out = out.replace("42", "age bucket 40-49")
        out = out.replace("Wien", "urban area")
        return out

    import hybrid_demo.edge.redaction_agent as redaction_agent

    monkeypatch.setattr(redaction_agent, "_redact_segment_with_slm", fake_slm)

    out = redact(_state(text))
    redacted_text = out.redacted_segments[0].text
    assert "Anna Müller" not in redacted_text
    assert "[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]" in redacted_text
    assert "age bucket 40-49" in redacted_text
    assert "urban area" in redacted_text
    assert "Atemnot" in redacted_text


def test_redacts_repeated_name_mentions_and_keeps_timestamp_context(monkeypatch):
    text = (
        "Darf ich erfahren, wie Sie heißen? Paul Gerster. "
        "Herr Gerster, wann hat das angefangen? Gestern gegen elf morgens?"
    )
    entities = [
        Entity(type="PERSON_NAME", value="Paul Gerster", placeholder="[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]"),
        Entity(type="TIMESTAMP", value="gestern gegen elf morgens", placeholder="[TIMESTAMP]"),
    ]

    def fake_slm(segment: str, _entities: list[Entity]) -> str:
        out = segment
        out = out.replace("Paul Gerster", "[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]")
        out = out.replace("Herr Gerster", "[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]")
        return out

    import hybrid_demo.edge.redaction_agent as redaction_agent

    monkeypatch.setattr(redaction_agent, "_redact_segment_with_slm", fake_slm)

    out = redact(_state_with_entities(text, entities))
    redacted_text = out.redacted_segments[0].text

    assert "Paul Gerster" not in redacted_text
    assert "Herr Gerster" not in redacted_text
    assert redacted_text.count("[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]") >= 2
    assert "Gestern gegen elf morgens" in redacted_text
    assert "[TIMESTAMP]" not in redacted_text
