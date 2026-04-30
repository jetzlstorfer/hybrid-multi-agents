from __future__ import annotations

from hybrid_demo.contracts import CloudResult, Explanation, WorkflowState
from hybrid_demo.edge.rehydration_agent import rehydrate


def test_rehydrate_injects_display_name_from_vault(monkeypatch):
    state = WorkflowState(workflow_id="wf_test")
    state.explanation = Explanation(
        workflow_id="wf_test",
        summary="summary text",
        clinical_reasoning=[],
        suggested_next_steps=[],
    )
    state.cloud_result = CloudResult(
        workflow_id="wf_test",
        possible_condition_categories=[],
        recommended_follow_up_questions=["Q1", "Q2"],
        red_flags=[],
        limitations=[],
    )

    def fake_reveal(_wf: str, key: str):
        if key == "[PATIENT_FIRST_NAME]":
            return "Anna"
        if key == "[PATIENT_LAST_NAME]":
            return "Mueller"
        return None

    monkeypatch.setattr(
        "hybrid_demo.edge.rehydration_agent.vault.reveal", fake_reveal)

    out = rehydrate(state)
    assert out.patient_display_name == "Anna Mueller"
    assert out.summary_for_clinician == "summary text"
    assert out.suggested_questions == ["Q1", "Q2"]
