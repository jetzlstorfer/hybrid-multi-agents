"""Cloud research agent.

Calls a Microsoft Foundry-hosted LLM via the Microsoft Agent Framework. Input
is a validated ``HandoverPackage`` (no identifiers). Output is a structured
``CloudResult``. No RAG: a single grounded LLM call is enough to make the
architectural point of the demo, and avoids stage-time vector-DB plumbing.
"""

from __future__ import annotations

import json

from .. import runtime, vault
from ..contracts import (
    CloudResult,
    ConditionCategory,
    HandoverPackage,
    WorkflowState,
)
from ..telemetry import traced_step


_RESEARCH_SYSTEM_PROMPT = """\
You are a clinical research assistant. You receive ONLY a pseudonymised
handover package - never patient identity. Do not request identity. Do not
invent clinical facts that are not in the input. Output strict JSON.

Schema:
{
  "possible_condition_categories": [
    {
      "category": "string",
      "examples": ["string"],
      "reasoning": "string",
      "urgency": "low" | "medium" | "high"
    }
  ],
  "recommended_follow_up_questions": ["string"],
  "red_flags": ["string"],
  "limitations": ["string"]
}
"""


@traced_step(
    name="cloud.research",
    runtime_location="cloud",
    input_classification="cloud_handover_package",
    output_classification="cloud_research_result",
    model_role="cloud.research",
)
async def research(state: WorkflowState) -> CloudResult:
    handover: HandoverPackage = state.handover  # type: ignore[assignment]

    token = vault.cloud_context()
    try:
        client = runtime.get_cloud_chat_client("cloud.research")

        from agent_framework import Agent

        agent = Agent(
            client=client,
            name="CloudResearchAgent",
            instructions=_RESEARCH_SYSTEM_PROMPT,
        )
        result = await agent.run(handover.model_dump_json())
        text = str(result)
        # Best effort to find the JSON block.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object in cloud response")
        parsed = json.loads(text[start : end + 1])

        return CloudResult(
            workflow_id=state.workflow_id,
            possible_condition_categories=[
                ConditionCategory(**c)
                for c in parsed.get("possible_condition_categories", [])
            ],
            recommended_follow_up_questions=parsed.get(
                "recommended_follow_up_questions", []
            ),
            red_flags=parsed.get("red_flags", []),
            limitations=parsed.get("limitations", []),
        )
    finally:
        vault.reset_cloud_context(token)
