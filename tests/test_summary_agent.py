from __future__ import annotations

from hybrid_demo.contracts import RedactedTranscript, TranscriptSegment, WorkflowState
from hybrid_demo.edge.summary_agent import summarise


async def test_summary_tolerates_invalid_symptom_items(monkeypatch):
    state = WorkflowState(workflow_id="wf_test")
    state.redacted = RedactedTranscript(
        workflow_id="wf_test",
        redacted_transcript_id="rtr_wf_test",
        redacted_segments=[TranscriptSegment(
            speaker="unknown", text="Patient reports chest pain")],
    )

    async def fake_slm_summary(_text: str):
        return {
            "patient_context": {},
            "chief_complaint": "chest pain",
            "symptoms": [
                {"name": None, "duration": "2 days"},
                {"name": "chest pain", "duration": "2 days",
                    "associated_symptoms": [None, "nausea"]},
                "not-a-dict",
            ],
            "known_medications": ["aspirin", None],
            "known_conditions": [None, "hypertension"],
            "negative_findings": ["no fever", None],
            "uncertainties": [None],
        }

    monkeypatch.setattr(
        "hybrid_demo.edge.summary_agent._slm_summary", fake_slm_summary)

    out = await summarise(state)

    assert out.chief_complaint == "chest pain"
    assert len(out.symptoms) == 1
    assert out.symptoms[0].name == "chest pain"
    assert out.symptoms[0].associated_symptoms == ["nausea"]
    assert out.known_medications == ["aspirin"]
    assert out.known_conditions == ["hypertension"]
    assert out.negative_findings == ["no fever"]
    assert out.uncertainties == []
