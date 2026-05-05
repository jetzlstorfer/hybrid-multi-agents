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
import aiohttp
import logging
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from . import config, policy, vault
from .cloud.explanation_agent import explain
from .cloud.research_agent import research
from .contracts import (
    CloudExecutionResponse,
    PolicyDecision,
    Transcript,
    TranscriptSegment,
    WorkflowState,
)
from .edge.pii_agent import detect_pii
from .edge.redaction_agent import redact
from .edge.rehydration_agent import rehydrate
from .edge.summary_agent import summarise
from .edge.transcription_agent import transcribe
from .telemetry import inject_trace_context, traced_step, tracer

_log = logging.getLogger(__name__)


@dataclass
class StageEvent:
    """Single pipeline-stage outcome streamed to clients."""

    stage: str
    payload: dict


def _progress(stage: str, message: str, runtime: str) -> StageEvent:
    return StageEvent(
        "progress",
        {
            "stage": stage,
            "message": message,
            "runtime": runtime,
        },
    )


async def _run_stage_with_timeout(
    func,
    state: WorkflowState,
    *,
    timeout_seconds: float,
    stage_name: str,
):
    start = asyncio.get_running_loop().time()
    _log.info(
        "Workflow %s: stage '%s' started (timeout=%ss)",
        state.workflow_id,
        stage_name,
        int(timeout_seconds),
    )
    try:
        result = await asyncio.wait_for(asyncio.to_thread(func, state), timeout=timeout_seconds)
        elapsed = asyncio.get_running_loop().time() - start
        _log.info(
            "Workflow %s: stage '%s' finished in %.1fs",
            state.workflow_id,
            stage_name,
            elapsed,
        )
        return result
    except TimeoutError as exc:
        elapsed = asyncio.get_running_loop().time() - start
        _log.warning(
            "Workflow %s: stage '%s' timed out after %.1fs (limit=%ss)",
            state.workflow_id,
            stage_name,
            elapsed,
            int(timeout_seconds),
        )
        raise TimeoutError(
            f"Stage '{stage_name}' timed out after {int(timeout_seconds)}s. "
            "Please retry with a shorter audio file or check local model runtime health."
        ) from exc


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


async def _run_remote_cloud_pipeline(
    state: WorkflowState,
    *,
    timeout_seconds: float,
) -> CloudExecutionResponse:
    cloud_backend_url = config.cloud_backend_url()
    if not cloud_backend_url:
        raise RuntimeError("HYBRID_DEMO_CLOUD_BACKEND_URL is not configured")

    endpoint = f"{cloud_backend_url.rstrip('/')}/api/cloud-run"
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = inject_trace_context({})
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            endpoint,
            json=state.handover.model_dump(),
            headers=headers,
        ) as response:
            if response.status >= 400:
                raw = await response.text()
                message: str | None = None
                try:
                    try:
                        payload = await response.json(content_type=None)
                    except TypeError:
                        payload = await response.json()
                    if isinstance(payload, dict):
                        detail = payload.get("detail")
                        if detail is not None:
                            message = str(detail)
                except Exception:
                    pass

                if message is None and raw:
                    # Keep error concise while still exposing ingress/router cause.
                    message = raw.strip().replace("\n", " ")[:240]

                raise RuntimeError(
                    f"Remote cloud backend returned HTTP {response.status}"
                    + (f": {message}" if message else "")
                )

            try:
                payload = await response.json(content_type=None)
            except TypeError:
                payload = await response.json()
    return CloudExecutionResponse.model_validate(payload)


async def run_workflow(
    audio_uri: str | None = None,
    *,
    language_hint: str | None = "de-AT",
    force_violation: bool = False,
    workflow_id: str | None = None,
    transcript_text: str | None = None,
) -> AsyncIterator[StageEvent]:
    import os
    # Edge stages (local SLM inference) need more time for CPU-based models.
    # Increase via HYBRID_DEMO_EDGE_TIMEOUT_SECONDS if needed (default 300s).
    edge_stage_timeout = float(os.environ.get(
        "HYBRID_DEMO_EDGE_TIMEOUT_SECONDS", "300"))
    # PII/entity extraction can be the slowest local SLM stage for long
    # transcripts. Allow a dedicated timeout override for this stage.
    pii_stage_timeout = float(os.environ.get(
        "HYBRID_DEMO_PII_TIMEOUT_SECONDS", "600"))
    cloud_stage_timeout = float(os.environ.get(
        "HYBRID_DEMO_CLOUD_TIMEOUT_SECONDS", "180"))
    # Transcription can take much longer than SLM stages for long audio files.
    # Default 900s to handle ~16-min files on CPU with the small model.
    # Override with HYBRID_DEMO_TRANSCRIPTION_TIMEOUT_SECONDS.
    transcription_timeout = float(os.environ.get(
        "HYBRID_DEMO_TRANSCRIPTION_TIMEOUT_SECONDS", "900"))
    """Run the full pipeline and yield stage events as they complete."""
    workflow_id = workflow_id or f"wf_{uuid.uuid4().hex[:8]}"
    state = WorkflowState(
        workflow_id=workflow_id,
        audio_uri=audio_uri,
        language_hint=language_hint,
        force_violation=force_violation,
        transcript_text=transcript_text,
    )

    with tracer().start_as_current_span("workflow") as span:
        span.set_attribute("workflow.id", workflow_id)
        try:
            # 1. Transcription
            if transcript_text:
                yield _progress("transcript", "Using provided transcript", "edge")
                state.transcript = Transcript(
                    workflow_id=state.workflow_id,
                    transcript_id=f"tr_{state.workflow_id}",
                    language=state.language_hint or "de-AT",
                    segments=[TranscriptSegment(
                        speaker="unknown", text=transcript_text)],
                )
            else:
                yield _progress("transcript", "Transcribing audio on edge model", "edge")
                state.transcript = await _run_stage_with_timeout(
                    transcribe,
                    state,
                    timeout_seconds=transcription_timeout,
                    stage_name="transcript",
                )
            yield StageEvent("transcript", state.transcript.model_dump())

            # 2. PII detection
            yield _progress("entities", "Detecting sensitive entities on edge SLM", "edge")
            state.sensitivity = await _run_stage_with_timeout(
                detect_pii,
                state,
                timeout_seconds=pii_stage_timeout,
                stage_name="entities",
            )
            yield StageEvent("entities", state.sensitivity.model_dump())

            # 3. Redaction
            yield _progress("redacted", "Redacting transcript on edge SLM", "edge")
            state.redacted = await _run_stage_with_timeout(
                redact,
                state,
                timeout_seconds=edge_stage_timeout,
                stage_name="redacted",
            )
            yield StageEvent("redacted", state.redacted.model_dump())

            # 4. Summary
            yield _progress("handover", "Building cloud handover package on edge SLM", "edge")
            state.handover = await _run_stage_with_timeout(
                summarise,
                state,
                timeout_seconds=edge_stage_timeout,
                stage_name="handover",
            )
            yield StageEvent("handover", state.handover.model_dump())

            # 5. Policy gate
            yield _progress("policy_gate", "Evaluating cloud policy gate", "gate")
            state.policy = await _run_stage_with_timeout(
                policy_gate,
                state,
                timeout_seconds=edge_stage_timeout,
                stage_name="policy_gate",
            )
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
            yield _progress("research", "Running cloud research agent", "cloud")
            if config.cloud_backend_url():
                try:
                    remote_result = await _run_remote_cloud_pipeline(
                        state,
                        timeout_seconds=cloud_stage_timeout,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"Stage 'research' timed out after {int(cloud_stage_timeout)}s. "
                        "Please verify remote cloud backend connectivity and retry."
                    ) from exc
                state.cloud_result = remote_result.cloud_result
                state.explanation = remote_result.explanation
            else:
                try:
                    state.cloud_result = await asyncio.wait_for(research(state), timeout=cloud_stage_timeout)
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"Stage 'research' timed out after {int(cloud_stage_timeout)}s. "
                        "Please verify cloud model connectivity and retry."
                    ) from exc
            yield StageEvent("research", state.cloud_result.model_dump())

            # 7. Cloud explanation
            yield _progress("explanation", "Generating cloud explanation", "cloud")
            if state.explanation is None:
                try:
                    state.explanation = await asyncio.wait_for(
                        explain(state),
                        timeout=cloud_stage_timeout,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"Stage 'explanation' timed out after {int(cloud_stage_timeout)}s. "
                        "Please verify cloud model connectivity and retry."
                    ) from exc
            yield StageEvent("explanation", state.explanation.model_dump())

            # 8. Rehydration
            yield _progress("final", "Rehydrating local placeholders", "edge")
            state.final = await _run_stage_with_timeout(
                rehydrate,
                state,
                timeout_seconds=edge_stage_timeout,
                stage_name="final",
            )
            yield StageEvent("final", state.final.model_dump())
        finally:
            vault.forget(workflow_id)
