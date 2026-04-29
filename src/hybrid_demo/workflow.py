"""Pipeline orchestration.

Pragmatic choice: instead of constructing an Agent Framework Workflow object
just to draw a diagram, we run the eight steps as a sequence of executor
functions that share a typed :class:`WorkflowState` and yield stage events as
they complete. This is exactly what the AG-UI server iterates over to stream
typed panel updates to the browser, and what the CLI prints to stdout.

The policy gate is its own step, executed *before* any cloud call. If
``cloud_allowed=False`` the pipeline short-circuits with a ``policy.gate``
event and the cloud steps are skipped.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from . import policy, vault
from .cloud.explanation_agent import explain
from .cloud.research_agent import research
from .contracts import (
    HandoverPackage,
    PolicyDecision,
    WorkflowState,
)
from .edge.pii_agent import detect_pii
from .edge.redaction_agent import redact
from .edge.rehydration_agent import rehydrate
from .edge.summary_agent import summarise
from .edge.transcription_agent import transcribe
from .telemetry import traced_step, tracer

_log = logging.getLogger(__name__)


@dataclass
class StageEvent:
    """Single pipeline-stage outcome streamed to clients."""

    stage: str
    payload: dict



@traced_step(
    name="policy.gate",
    runtime_location="edge",
    input_classification="cloud_handover_package",
    output_classification="policy_decision",
)
def policy_gate(state: WorkflowState) -> PolicyDecision:
    handover = state.handover
    if handover is None:
        return PolicyDecision(cloud_allowed=False, violations=["No handover package"])

    payload: dict
    if state.force_violation:
        # Build an obviously-spoiled dict that mirrors the README "Should Fail" example.
        data = handover.model_dump()
        data["patient_context"] = {
            **(data.get("patient_context") or {}),
            "name": "Anna Müller",
            "date_of_birth": "1983-04-18",
            "address": "Example Street 12, Vienna",
        }
        payload = data
    else:
        payload = handover.model_dump()

    return policy.validate_handover(payload)


async def run_workflow(
    audio_uri: str | None = None,
    *,
    language_hint: str | None = "de-AT",
    force_violation: bool = False,
    workflow_id: str | None = None,
) -> AsyncIterator[StageEvent]:
    """Run the full pipeline and yield stage events as they complete."""
    workflow_id = workflow_id or f"wf_{uuid.uuid4().hex[:8]}"
    state = WorkflowState(
        workflow_id=workflow_id,
        audio_uri=audio_uri,
        language_hint=language_hint,
        force_violation=force_violation,
    )

    with tracer().start_as_current_span("workflow") as span:
        span.set_attribute("workflow.id", workflow_id)
        try:
            # 1. Transcription
            state.transcript = await asyncio.to_thread(transcribe, state)
            yield StageEvent("transcript", state.transcript.model_dump())

            # 2. PII detection
            state.sensitivity = await asyncio.to_thread(detect_pii, state)
            yield StageEvent("entities", state.sensitivity.model_dump())

            # 3. Redaction
            state.redacted = await asyncio.to_thread(redact, state)
            yield StageEvent("redacted", state.redacted.model_dump())

            # 4. Summary
            state.handover = await asyncio.to_thread(summarise, state)
            yield StageEvent("handover", state.handover.model_dump())

            # 5. Policy gate
            state.policy = await asyncio.to_thread(policy_gate, state)
            yield StageEvent("policy_gate", state.policy.model_dump())

            if not state.policy.cloud_allowed:
                _log.info(
                    "Policy gate blocked workflow %s: %s",
                    workflow_id,
                    state.policy.violations,
                )
                yield StageEvent(
                    "blocked",
                    {
                        "workflow_id": workflow_id,
                        "violations": state.policy.violations,
                    },
                )
                return

            # 6. Cloud research
            state.cloud_result = await research(state)
            yield StageEvent("research", state.cloud_result.model_dump())

            # 7. Cloud explanation
            state.explanation = await explain(state)
            yield StageEvent("explanation", state.explanation.model_dump())

            # 8. Rehydration
            state.final = await asyncio.to_thread(rehydrate, state)
            yield StageEvent("final", state.final.model_dump())
        finally:
            vault.forget(workflow_id)
