from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

from hybrid_demo import config
from hybrid_demo.contracts import (
    CloudExecutionResponse,
    CloudResult,
    Explanation,
    FinalResponse,
    HandoverPackage,
    RedactedTranscript,
    SensitivityReport,
    TranscriptSegment,
)
from hybrid_demo.ag_ui_server import cloud_run
from hybrid_demo.telemetry import inject_trace_context, use_extracted_context
from hybrid_demo.workflow import run_workflow


class _FakeResponse:
    status = 200

    async def json(self):
        return {
            "workflow_id": "wf_test",
            "cloud_result": {
                "workflow_id": "wf_test",
                "possible_condition_categories": [],
                "recommended_follow_up_questions": [],
                "red_flags": [],
                "limitations": [],
            },
            "explanation": {
                "workflow_id": "wf_test",
                "summary": "ok",
                "clinical_reasoning": [],
                "suggested_next_steps": [],
                "safety_note": "Decision support only.",
            },
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeClientSession:
    def __init__(self, *, timeout):
        self.timeout = timeout
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, json, headers):
        self.calls.append({"url": url, "json": json, "headers": dict(headers)})
        return _FakeResponse()


class _RecordedSpan:
    def __init__(self, name, *, kind):
        self.name = name
        self.kind = kind
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _RecordedSpanContext:
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordedTracer:
    def __init__(self):
        self.spans: list[_RecordedSpan] = []

    def start_as_current_span(self, name, *, kind):
        span = _RecordedSpan(name, kind=kind)
        self.spans.append(span)
        return _RecordedSpanContext(span)


def test_cloud_backend_url_reads_env(monkeypatch):
    monkeypatch.setenv("HYBRID_DEMO_CLOUD_BACKEND_URL", "https://cloud.example")
    assert config.cloud_backend_url() == "https://cloud.example"


@pytest.mark.asyncio
async def test_remote_cloud_pipeline_injects_trace_headers(monkeypatch):
    from hybrid_demo.contracts import HandoverPackage, WorkflowState
    from hybrid_demo.workflow import _run_remote_cloud_pipeline

    monkeypatch.setattr("hybrid_demo.workflow.config.cloud_backend_url", lambda: "https://cloud.example")

    monkeypatch.setattr(
        "hybrid_demo.workflow.aiohttp.ClientSession",
        lambda *, timeout: _FakeClientSession(timeout=timeout),
    )

    parent_context = SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
        trace_state=TraceState(),
    )
    token = attach(trace.set_span_in_context(NonRecordingSpan(parent_context)))
    try:
        captured: dict = {}

        def make_session(*, timeout):
            fake = _FakeClientSession(timeout=timeout)
            captured["session"] = fake
            return fake

        monkeypatch.setattr("hybrid_demo.workflow.aiohttp.ClientSession", make_session)

        response = await _run_remote_cloud_pipeline(
            WorkflowState(
                workflow_id="wf_test",
                handover=HandoverPackage(workflow_id="wf_test"),
            ),
            timeout_seconds=180,
        )
    finally:
        detach(token)

    assert response.workflow_id == "wf_test"
    headers = captured["session"].calls[0]["headers"]
    assert "traceparent" in headers
    assert headers["traceparent"].startswith("00-1234567890abcdef1234567890abcdef-1234567890abcdef-")


def test_extracted_trace_context_restores_remote_parent():
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    parent_context = SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
        trace_state=TraceState(),
    )
    token = attach(trace.set_span_in_context(NonRecordingSpan(parent_context)))
    try:
        headers = inject_trace_context({})
    finally:
        detach(token)

    with use_extracted_context(headers):
        with tracer.start_as_current_span("cloud-step") as span:
            context = span.get_span_context()
            assert context.trace_id == parent_context.trace_id
            assert span.parent is not None
            assert span.parent.span_id == parent_context.span_id


@pytest.mark.asyncio
async def test_cloud_run_creates_server_span(monkeypatch):
    tracer = _RecordedTracer()
    monkeypatch.setattr("hybrid_demo.ag_ui_server.telemetry.tracer", lambda: tracer)
    monkeypatch.setattr("hybrid_demo.ag_ui_server.config.deployment_mode", lambda: "cloud")

    async def fake_research(state):
        return CloudResult(
            workflow_id=state.workflow_id,
            possible_condition_categories=[],
            recommended_follow_up_questions=[],
            red_flags=[],
            limitations=[],
        )

    async def fake_explain(state):
        return Explanation(
            workflow_id=state.workflow_id,
            summary="summary",
            clinical_reasoning=[],
            suggested_next_steps=[],
            safety_note="Decision support only.",
        )

    monkeypatch.setattr("hybrid_demo.ag_ui_server.research", fake_research)
    monkeypatch.setattr("hybrid_demo.ag_ui_server.explain", fake_explain)

    response = await cloud_run(
        SimpleNamespace(headers={"traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"}),
        HandoverPackage(workflow_id="wf_test"),
    )

    assert response.workflow_id == "wf_test"
    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "http.cloud_run"
    assert tracer.spans[0].kind == trace.SpanKind.SERVER
    assert tracer.spans[0].attributes["workflow.id"] == "wf_test"
    assert tracer.spans[0].attributes["http.route"] == "/api/cloud-run"


@pytest.mark.asyncio
async def test_workflow_can_delegate_cloud_stages(monkeypatch):
    remote_response = CloudExecutionResponse(
        workflow_id="wf_test",
        cloud_result=CloudResult(
            workflow_id="wf_test",
            possible_condition_categories=[],
            recommended_follow_up_questions=["Seit wann genau?"],
            red_flags=[],
            limitations=[],
        ),
        explanation=Explanation(
            workflow_id="wf_test",
            summary="Remote cloud summary",
            clinical_reasoning=["Remote cloud reasoning"],
            suggested_next_steps=["Neurologische Untersuchung"],
            safety_note="Decision support only.",
        ),
    )

    monkeypatch.setattr("hybrid_demo.workflow.config.cloud_backend_url", lambda: "https://cloud.example")

    async def fake_remote(state, *, timeout_seconds):
        assert state.handover is not None
        assert timeout_seconds == 180.0
        return remote_response

    monkeypatch.setattr("hybrid_demo.workflow._run_remote_cloud_pipeline", fake_remote)
    monkeypatch.setattr(
        "hybrid_demo.workflow.detect_pii",
        lambda _state: SensitivityReport(workflow_id="wf_test", entities=[]),
    )
    monkeypatch.setattr(
        "hybrid_demo.workflow.redact",
        lambda _state: RedactedTranscript(
            workflow_id="wf_test",
            redacted_transcript_id="rtr_wf_test",
            redacted_segments=[TranscriptSegment(speaker="unknown", text="[REDACTED]")],
        ),
    )
    monkeypatch.setattr(
        "hybrid_demo.workflow.summarise",
        lambda _state: HandoverPackage(
            workflow_id="wf_test",
            chief_complaint="Nackenschmerz",
            forbidden_fields_removed=True,
        ),
    )
    monkeypatch.setattr(
        "hybrid_demo.workflow.rehydrate",
        lambda state: FinalResponse(
            workflow_id=state.workflow_id,
            summary_for_clinician=state.explanation.summary,
            suggested_questions=state.cloud_result.recommended_follow_up_questions,
        ),
    )

    events = []
    async for event in run_workflow(workflow_id="wf_test", transcript_text="Symptomtext"):
        events.append(event)

    research_event = next(event for event in events if event.stage == "research")
    explanation_event = next(event for event in events if event.stage == "explanation")
    final_event = next(event for event in events if event.stage == "final")

    assert research_event.payload["recommended_follow_up_questions"] == ["Seit wann genau?"]
    assert explanation_event.payload["summary"] == "Remote cloud summary"
    assert final_event.payload["summary_for_clinician"] == "Remote cloud summary"