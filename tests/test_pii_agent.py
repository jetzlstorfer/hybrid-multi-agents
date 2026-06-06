from __future__ import annotations

from hybrid_demo.contracts import Entity, Transcript, TranscriptSegment, WorkflowState
from hybrid_demo.edge.pii_agent import detect_pii


async def test_detect_pii_extracts_entities_without_runtime(monkeypatch):
    state = WorkflowState(workflow_id="wf_test")
    state.transcript = Transcript(
        workflow_id="wf_test",
        transcript_id="tr_test",
        language="de",
        segments=[
            TranscriptSegment(
                speaker="unknown",
                text="Mein Name ist Anna Mueller und ich wohne in Wien.",
            )
        ],
    )

    async def fake_slm_entities(_text):
        return [
            Entity(
                type="PERSON_NAME",
                value="Anna Mueller",
                placeholder="[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]",
            ),
            Entity(
                type="LOCATION",
                value="Wien",
                placeholder="[LOCATION]",
            ),
        ]

    monkeypatch.setattr(
        "hybrid_demo.edge.pii_agent._slm_entities", fake_slm_entities)

    out = await detect_pii(state)
    assert len(out.entities) == 2
    assert {e.type for e in out.entities} == {"PERSON_NAME", "LOCATION"}


async def test_detect_pii_handles_empty_entities(monkeypatch):
    state = WorkflowState(workflow_id="wf_test")
    state.transcript = Transcript(
        workflow_id="wf_test",
        transcript_id="tr_test",
        language="de",
        segments=[TranscriptSegment(speaker="unknown", text="kein pii")],
    )

    async def fake_slm_entities(_text):
        return []

    monkeypatch.setattr(
        "hybrid_demo.edge.pii_agent._slm_entities", fake_slm_entities)

    out = await detect_pii(state)
    assert out.entities == []


async def test_pii_chunk_retry_on_cancellation(monkeypatch):
    import hybrid_demo.edge.pii_agent as pii_agent

    class DummyAgent:
        pass

    # Simulate runtime cancellation for long chunks only.
    async def fake_complete_chat_json(_agent, user_text):
        if len(user_text) > 60:
            raise RuntimeError(
                "Error during chat completion: Operation was cancelled")
        return {"entities": [{"type": "LOCATION", "value": "Wien"}]}

    monkeypatch.setattr(pii_agent, "_complete_chat_json",
                        fake_complete_chat_json)

    chunk = "Wien " * 40  # Long enough to trigger splitting in the fake above.
    entities = await pii_agent._slm_entities_for_chunk(
        DummyAgent(),
        chunk,
        min_chunk_chars=40,
    )

    assert any(e.type == "LOCATION" and e.value == "Wien" for e in entities)
