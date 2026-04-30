from __future__ import annotations

import asyncio
import types

from hybrid_demo.contracts import (
    CloudResult,
    HandoverPackage,
    WorkflowState,
)
from hybrid_demo.cloud.explanation_agent import explain
from hybrid_demo.cloud.research_agent import research


class _DummyAgent:
    def __init__(self, client, name: str, instructions: str):
        self.client = client
        self.name = name
        self.instructions = instructions

    async def run(self, payload: str):
        # Simulate framework output with prose around JSON.
        if self.name == "CloudResearchAgent":
            return (
                "prefix {\"possible_condition_categories\": "
                "[{\"category\":\"respiratory\",\"examples\":[\"asthma\"],"
                "\"reasoning\":\"wheeze\",\"urgency\":\"medium\"}],"
                "\"recommended_follow_up_questions\":[\"since when?\"],"
                "\"red_flags\":[],\"limitations\":[]} suffix"
            )
        return (
            "prefix {\"summary\":\"Likely non-emergent\","
            "\"clinical_reasoning\":[\"No red flags\"],"
            "\"suggested_next_steps\":[\"GP follow-up\"],"
            "\"safety_note\":\"Not a diagnosis\"} suffix"
        )


def test_research_agent_isolated(monkeypatch):
    state = WorkflowState(workflow_id="wf_test")
    state.handover = HandoverPackage(workflow_id="wf_test")

    monkeypatch.setattr(
        "hybrid_demo.cloud.research_agent.runtime.get_cloud_chat_client", lambda _role: object())
    monkeypatch.setattr(
        "hybrid_demo.cloud.research_agent.vault.cloud_context", lambda: "tok")
    monkeypatch.setattr(
        "hybrid_demo.cloud.research_agent.vault.reset_cloud_context", lambda _tok: None)
    monkeypatch.setitem(__import__("sys").modules, "agent_framework",
                        types.SimpleNamespace(Agent=_DummyAgent))

    out = asyncio.run(research(state))
    assert out.workflow_id == "wf_test"
    assert out.possible_condition_categories[0].category == "respiratory"
    assert out.recommended_follow_up_questions == ["since when?"]


def test_explanation_agent_isolated(monkeypatch):
    state = WorkflowState(workflow_id="wf_test")
    state.cloud_result = CloudResult(
        workflow_id="wf_test",
        possible_condition_categories=[],
        recommended_follow_up_questions=[],
        red_flags=[],
        limitations=[],
    )

    monkeypatch.setattr(
        "hybrid_demo.cloud.explanation_agent.runtime.get_cloud_chat_client", lambda _role: object())
    monkeypatch.setattr(
        "hybrid_demo.cloud.explanation_agent.vault.cloud_context", lambda: "tok")
    monkeypatch.setattr(
        "hybrid_demo.cloud.explanation_agent.vault.reset_cloud_context", lambda _tok: None)
    monkeypatch.setitem(__import__("sys").modules, "agent_framework",
                        types.SimpleNamespace(Agent=_DummyAgent))

    out = asyncio.run(explain(state))
    assert out.workflow_id == "wf_test"
    assert out.summary == "Likely non-emergent"
    assert out.suggested_next_steps == ["GP follow-up"]
