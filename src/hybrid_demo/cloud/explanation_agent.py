"""Cloud explanation agent.

Turns the structured ``CloudResult`` into a clinician-facing ``Explanation``
with a clear safety note and separated reasoning vs uncertainty.
"""

from __future__ import annotations

import json

from .. import runtime, vault
from ..contracts import CloudResult, Explanation, WorkflowState
from ..telemetry import traced_step


_EXPLANATION_SYSTEM_PROMPT = """\
You are a clinical explanation agent. Input is a structured CloudResult.
Output strict JSON. Avoid diagnostic finality. Separate evidence from
uncertainty. Never reference patient identity.

Schema:
{
  "summary": "string",
  "clinical_reasoning": ["string"],
  "suggested_next_steps": ["string"],
  "safety_note": "string"
}
"""


@traced_step(
    name="cloud.explanation",
    runtime_location="cloud",
    input_classification="cloud_research_result",
    output_classification="cloud_explanation",
    model_role="cloud.explanation",
)
async def explain(state: WorkflowState) -> Explanation:
    cloud_result: CloudResult = state.cloud_result  # type: ignore[assignment]

    token = vault.cloud_context()
    try:
        client = runtime.get_cloud_chat_client("cloud.explanation")

        from agent_framework import Agent

        agent = Agent(
            client=client,
            name="CloudExplanationAgent",
            instructions=_EXPLANATION_SYSTEM_PROMPT,
        )
        result = await agent.run(cloud_result.model_dump_json())
        text = str(result)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object in cloud response")
        parsed = json.loads(text[start : end + 1])

        return Explanation(
            workflow_id=state.workflow_id,
            summary=parsed.get("summary", ""),
            clinical_reasoning=parsed.get("clinical_reasoning", []),
            suggested_next_steps=parsed.get("suggested_next_steps", []),
            safety_note=parsed.get(
                "safety_note",
                "This output is decision support only and does not replace clinical judgment.",
            ),
        )
    finally:
        vault.reset_cloud_context(token)
