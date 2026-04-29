"""OpenTelemetry setup + the ``traced_step`` decorator.

Spans never carry raw payload bodies. Only classifications, decisions, model
identifiers, and the workflow id are recorded as attributes. This is how the
demo proves the data boundary visually in the Foundry project's Tracing tab.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
from typing import Any, Awaitable, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from . import config

_log = logging.getLogger(__name__)

_initialised = False
_provider: TracerProvider | None = None
SERVICE_NAME = "hybrid-demo"


def init_tracing() -> None:
    global _initialised, _provider
    if _initialised:
        return

    resource = Resource.create({"service.name": SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if conn:
        try:
            from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

            provider.add_span_processor(
                BatchSpanProcessor(AzureMonitorTraceExporter(connection_string=conn))
            )
            _log.info("Azure Monitor trace exporter enabled.")
        except Exception as exc:  # pragma: no cover - infra-dependent
            _log.warning("Could not enable Azure Monitor exporter: %s", exc)

    trace.set_tracer_provider(provider)
    _provider = provider
    _initialised = True


def shutdown_tracing() -> None:
    """Flush and shut down exporters so spans are not lost on process exit."""
    global _provider
    if _provider is None:
        return
    try:
        _provider.force_flush()
    except Exception:  # pragma: no cover - exporter dependent
        pass
    try:
        _provider.shutdown()
    except Exception:  # pragma: no cover - exporter dependent
        pass
    _provider = None


def tracer():
    init_tracing()
    return trace.get_tracer(SERVICE_NAME)


F = TypeVar("F", bound=Callable[..., Any])


def traced_step(
    name: str,
    *,
    runtime_location: str,
    input_classification: str,
    output_classification: str,
    model_role: str | None = None,
) -> Callable[[F], F]:
    """Decorator that wraps an executor function in a span with safe attributes.

    The decorated function must accept ``state: WorkflowState`` as either the
    first positional or a keyword argument so we can read ``workflow_id``.
    """

    def decorator(func: F) -> F:
        is_coro = inspect.iscoroutinefunction(func)

        def _obj_preview(obj: Any) -> str:
            if obj is None:
                return "none"
            if isinstance(obj, str):
                return f"text(len={len(obj)})"
            if isinstance(obj, (list, tuple, set)):
                return f"{type(obj).__name__}(len={len(obj)})"
            if isinstance(obj, dict):
                keys = sorted(str(k) for k in obj.keys())
                return f"dict(keys={keys[:8]}, total_keys={len(keys)})"

            parts = [f"type={type(obj).__name__}"]
            for attr in (
                "segments",
                "redacted_segments",
                "entities",
                "possible_condition_categories",
                "recommended_follow_up_questions",
                "red_flags",
                "limitations",
                "clinical_reasoning",
                "suggested_next_steps",
                "violations",
            ):
                value = getattr(obj, attr, None)
                if isinstance(value, list):
                    parts.append(f"{attr}={len(value)}")

            for attr in ("cloud_allowed", "force_violation"):
                value = getattr(obj, attr, None)
                if isinstance(value, bool):
                    parts.append(f"{attr}={value}")

            return "; ".join(parts)

        def _state_preview(state: Any) -> str:
            if state is None:
                return "state=none"

            workflow_id = getattr(state, "workflow_id", "unknown")
            stage_fields = []
            for field in (
                "transcript",
                "sensitivity",
                "redacted",
                "handover",
                "policy",
                "cloud_result",
                "explanation",
                "final",
            ):
                if getattr(state, field, None) is not None:
                    stage_fields.append(field)

            return (
                f"workflow_id={workflow_id}; "
                f"available_state={stage_fields}; "
                f"state_type={type(state).__name__}"
            )

        def _event_payload(text: str) -> str:
            # Foundry default query resolves message.content.text.value.
            return json.dumps(
                {
                    "message": {
                        "content": {
                            "text": {"value": text}
                        }
                    }
                },
                ensure_ascii=False,
            )

        def _emit_genai_event(span, event_name: str, text: str) -> None:
            span.add_event(
                event_name,
                {
                    "event.name": event_name,
                    "gen_ai.event.content": _event_payload(text),
                },
            )

        def _attrs(state) -> dict[str, Any]:
            workflow_id = getattr(state, "workflow_id", "unknown")
            attrs: dict[str, Any] = {
                "workflow.id": workflow_id,
                "agent.name": name,
                "runtime.location": runtime_location,
                "input.classification": input_classification,
                "output.classification": output_classification,
                "gen_ai.system": SERVICE_NAME,
                "gen_ai.provider.name": "hybrid-demo",
            }
            if model_role is not None:
                try:
                    spec = config.get_model(model_role)
                    attrs["model.name"] = spec.model
                    attrs["model.provider"] = spec.provider
                    attrs["gen_ai.provider.name"] = spec.provider
                    attrs["gen_ai.request.model"] = spec.model
                except Exception:  # pragma: no cover
                    attrs["model.name"] = "unknown"
            attrs["gen_ai.response.id"] = f"{workflow_id}:{name}"
            return attrs

        def _set_enrichment_attrs(span, state: Any, result: Any) -> None:
            """Attach compact, non-sensitive stage diagnostics."""
            if state is not None:
                transcript = getattr(state, "transcript", None)
                sensitivity = getattr(state, "sensitivity", None)
                redacted = getattr(state, "redacted", None)
                handover = getattr(state, "handover", None)
                policy_decision = getattr(state, "policy", None)

                if transcript is not None:
                    segments = getattr(transcript, "segments", None)
                    if isinstance(segments, list):
                        span.set_attribute("input.transcript.segment_count", len(segments))

                if sensitivity is not None:
                    entities = getattr(sensitivity, "entities", None)
                    if isinstance(entities, list):
                        span.set_attribute("input.sensitivity.entity_count", len(entities))

                if redacted is not None:
                    redacted_segments = getattr(redacted, "redacted_segments", None)
                    if isinstance(redacted_segments, list):
                        span.set_attribute("input.redaction.segment_count", len(redacted_segments))

                if handover is not None:
                    span.set_attribute("input.handover.present", True)

                if policy_decision is not None:
                    cloud_allowed = getattr(policy_decision, "cloud_allowed", None)
                    violations = getattr(policy_decision, "violations", None)
                    if isinstance(cloud_allowed, bool):
                        span.set_attribute("input.policy.cloud_allowed", cloud_allowed)
                    if isinstance(violations, list):
                        span.set_attribute("input.policy.violation_count", len(violations))

            if result is None:
                return

            for attr, key in (
                ("segments", "output.transcript.segment_count"),
                ("redacted_segments", "output.redaction.segment_count"),
                ("entities", "output.sensitivity.entity_count"),
                ("possible_condition_categories", "output.research.condition_count"),
                ("recommended_follow_up_questions", "output.research.follow_up_count"),
                ("red_flags", "output.research.red_flag_count"),
                ("limitations", "output.research.limitation_count"),
                ("clinical_reasoning", "output.explanation.reasoning_count"),
                ("suggested_next_steps", "output.explanation.next_step_count"),
                ("violations", "output.policy.violation_count"),
            ):
                value = getattr(result, attr, None)
                if isinstance(value, list):
                    span.set_attribute(key, len(value))

            cloud_allowed = getattr(result, "cloud_allowed", None)
            if isinstance(cloud_allowed, bool):
                span.set_attribute("output.policy.cloud_allowed", cloud_allowed)

        def _completion_text(result: Any) -> str:
            preview = _obj_preview(result)
            if name == "policy.gate":
                allowed = getattr(result, "cloud_allowed", None)
                violations = getattr(result, "violations", None)
                v_count = len(violations) if isinstance(violations, list) else "unknown"
                return (
                    f"policy decision: cloud_allowed={allowed}; violations={v_count}; "
                    f"summary=({preview})"
                )

            if name == "edge.pii":
                entities = getattr(result, "entities", None)
                count = len(entities) if isinstance(entities, list) else "unknown"
                return f"pii extraction completed: detected_entities={count}; summary=({preview})"

            if name == "edge.transcription":
                segments = getattr(result, "segments", None)
                count = len(segments) if isinstance(segments, list) else "unknown"
                return f"transcription completed: segments={count}; summary=({preview})"

            if name == "edge.redaction":
                redacted_segments = getattr(result, "redacted_segments", None)
                count = len(redacted_segments) if isinstance(redacted_segments, list) else "unknown"
                return f"redaction completed: redacted_segments={count}; summary=({preview})"

            if name == "edge.summary":
                return f"handover package prepared for cloud step; summary=({preview})"

            if name == "cloud.research":
                conditions = getattr(result, "possible_condition_categories", None)
                cond_count = len(conditions) if isinstance(conditions, list) else "unknown"
                return (
                    f"cloud research completed: condition_candidates={cond_count}; "
                    f"summary=({preview})"
                )

            if name == "cloud.explanation":
                return f"cloud explanation generated for clinician-facing output; summary=({preview})"

            if name == "edge.rehydration":
                return f"local rehydration completed for final response; summary=({preview})"

            return (
                f"completed {name}; output_classification={output_classification}; "
                f"output_summary=({preview})"
            )

        def _emit_flow_markers(span, state, *, completion: bool, result: Any = None) -> None:
            workflow_id = getattr(state, "workflow_id", "unknown")
            if not completion:
                _emit_genai_event(
                    span,
                    "gen_ai.system.message",
                    (
                        f"agent={name}; workflow_id={workflow_id}; runtime={runtime_location}; "
                        f"input_classification={input_classification}; output_classification={output_classification}"
                    ),
                )
                _emit_genai_event(
                    span,
                    "gen_ai.user.message",
                    f"execute {name}; input_summary=({_state_preview(state)})",
                )
                return

            completion_text = _completion_text(result)
            _emit_genai_event(span, "gen_ai.assistant.message", completion_text)
            _emit_genai_event(span, "gen_ai.choice", completion_text)

        if is_coro:

            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                state = kwargs.get("state") or (args[0] if args else None)
                with tracer().start_as_current_span(name) as span:
                    for k, v in _attrs(state).items():
                        span.set_attribute(k, v)
                    _emit_flow_markers(span, state, completion=False)
                    try:
                        result = await func(*args, **kwargs)  # type: ignore[misc]
                    except Exception:
                        span.set_attribute("gen_ai.response.finish_reasons", "error")
                        raise
                    _set_enrichment_attrs(span, state, result)
                    _emit_flow_markers(span, state, completion=True, result=result)
                    span.set_attribute("gen_ai.response.finish_reasons", "stop")
                    return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            state = kwargs.get("state") or (args[0] if args else None)
            with tracer().start_as_current_span(name) as span:
                for k, v in _attrs(state).items():
                    span.set_attribute(k, v)
                _emit_flow_markers(span, state, completion=False)
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    span.set_attribute("gen_ai.response.finish_reasons", "error")
                    raise
                _set_enrichment_attrs(span, state, result)
                _emit_flow_markers(span, state, completion=True, result=result)
                span.set_attribute("gen_ai.response.finish_reasons", "stop")
                return result

        return wrapper  # type: ignore[return-value]

    return decorator
