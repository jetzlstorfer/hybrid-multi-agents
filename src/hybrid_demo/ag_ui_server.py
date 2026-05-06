"""FastAPI server exposing the pipeline as AG-UI-style event streams.

The AG-UI protocol is event-based and transports over SSE/HTTP. To keep the
demo readable we emit a small, custom event type per pipeline stage
(``stage.transcript``, ``stage.entities``, ``stage.policy_gate`` …). Each
event payload is the typed contract for that stage, which the Next.js
frontend renders as its corresponding panel. CopilotKit on the client side
treats these as custom events while still using its standard SSE transport.

Two endpoints:

* ``POST /api/run``      - kick off a workflow run; returns ``{workflow_id}``.
* ``GET  /api/events/{id}`` - SSE stream of stage events for that run.
* ``POST /agui``          - AG-UI-compatible single-shot endpoint that streams
                            the same events under the agent-user interaction
                            protocol envelope.

A health endpoint ``/healthz`` is exposed for the cluster probe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from sse_starlette.sse import EventSourceResponse

from . import config, runtime, telemetry
from .cloud.explanation_agent import explain
from .cloud.research_agent import research
from .contracts import CloudExecutionResponse, HandoverPackage, WorkflowState
from .workflow import StageEvent, run_workflow

_log = logging.getLogger(__name__)

# In-memory queue of events per workflow run. Single-process; the demo runs
# one workflow at a time on stage so this is sufficient.
_queues: dict[str, asyncio.Queue] = {}
_SENTINEL = object()


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    telemetry.init_tracing()
    runtime.log_startup_diagnostics()
    yield
    telemetry.shutdown_tracing()


app = FastAPI(title="Hybrid Multi-Agent Demo", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_mode(*allowed: str) -> None:
    mode = config.deployment_mode()
    if mode not in allowed:
        raise HTTPException(
            status_code=404,
            detail=f"Endpoint not available in deployment mode '{mode}'",
        )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
async def api_status() -> JSONResponse:
    """Report edge model cache state. Used by the status page in the web UI."""
    if config.deployment_mode() == "cloud":
        return JSONResponse(
            {
                "overall": "ok",
                "mode": "cloud",
                "models": [],
                "detail": "Local model cache reporting is disabled in cloud deployment mode.",
            }
        )
    models = _local_model_cache_statuses()
    model_statuses = {m["status"] for m in models}
    sdk_ok = bool(models) and all(
        s in {"loaded", "cached", "not_cached"} for s in model_statuses)
    sdk_missing = "sdk_missing" in model_statuses
    overall = "ok" if sdk_ok else (
        "not_configured" if sdk_missing else "degraded")
    return JSONResponse({"overall": overall, "models": models})


def _local_model_cache_statuses() -> list[dict]:
    """Report cache/load status for edge models declared in models.yaml."""
    roles = [
        ("edge.transcription", "Edge transcription"),
        ("edge.slm", "Edge SLM"),
    ]

    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except ImportError:
        return [
            {
                "role": role,
                "label": label,
                "model": config.get_model(role).model,
                "status": "sdk_missing",
                "is_cached": False,
                "is_loaded": False,
                "path": None,
                "detail": "foundry-local-sdk is not installed in the backend environment",
            }
            for role, label in roles
        ]

    try:
        if FoundryLocalManager.instance is None:
            FoundryLocalManager.initialize(
                Configuration(app_name="hybrid_demo"))
        manager = FoundryLocalManager.instance
    except Exception as exc:
        return [
            {
                "role": role,
                "label": label,
                "model": config.get_model(role).model,
                "status": "error",
                "is_cached": False,
                "is_loaded": False,
                "path": None,
                "detail": f"Failed to initialize Foundry Local manager: {exc}",
            }
            for role, label in roles
        ]

    out: list[dict] = []
    for role, label in roles:
        spec = config.get_model(role)
        try:
            model = manager.catalog.get_model(spec.model)
            cached_attr = getattr(model, "is_cached", False)
            loaded_attr = getattr(model, "is_loaded", False)
            is_cached = bool(cached_attr() if callable(
                cached_attr) else cached_attr)
            is_loaded = bool(loaded_attr() if callable(
                loaded_attr) else loaded_attr)
            status = "loaded" if is_loaded else (
                "cached" if is_cached else "not_cached")
            path = None
            if is_cached:
                try:
                    get_path = getattr(model, "get_path", None)
                    if callable(get_path):
                        path = get_path()
                except Exception:
                    path = None
            out.append(
                {
                    "role": role,
                    "label": label,
                    "model": spec.model,
                    "status": status,
                    "is_cached": is_cached,
                    "is_loaded": is_loaded,
                    "path": path,
                    "detail": (
                        "Model is loaded in memory"
                        if is_loaded
                        else "Model is downloaded to local cache"
                        if is_cached
                        else "Model not downloaded yet"
                    ),
                }
            )
        except Exception as exc:
            out.append(
                {
                    "role": role,
                    "label": label,
                    "model": spec.model,
                    "status": "error",
                    "is_cached": False,
                    "is_loaded": False,
                    "path": None,
                    "detail": str(exc),
                }
            )

    return out


async def _drive_workflow(
    workflow_id: str,
    audio_uri: str | None,
    transcript_text: str | None,
    language_hint: str | None,
    force_violation: bool,
) -> None:
    queue = _queues[workflow_id]
    try:
        async for event in run_workflow(
            audio_uri=audio_uri,
            transcript_text=transcript_text,
            language_hint=language_hint,
            force_violation=force_violation,
            workflow_id=workflow_id,
        ):
            await queue.put(event)
    except Exception as exc:  # pragma: no cover
        _log.exception("Workflow %s failed", workflow_id)
        await queue.put(StageEvent("error", {"message": str(exc)}))
    finally:
        await queue.put(_SENTINEL)


@app.post("/api/run")
async def run(
    background_tasks: BackgroundTasks,
    audio: UploadFile | None = None,
    transcript: str = Form(""),
    language_hint: str = Form("de-AT"),
    force_violation: bool = Form(False),
) -> JSONResponse:
    _require_mode("edge")
    workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
    _queues[workflow_id] = asyncio.Queue()

    audio_uri: str | None = None
    transcript_text: str | None = None

    if transcript and transcript.strip():
        transcript_text = transcript.strip()
    elif audio is not None and audio.filename:
        path = f"/tmp/{workflow_id}_{audio.filename}"
        with open(path, "wb") as fh:
            fh.write(await audio.read())
        audio_uri = path
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Either an audio file or a transcript is required"},
        )

    background_tasks.add_task(
        _drive_workflow, workflow_id, audio_uri, transcript_text, language_hint, force_violation
    )
    return JSONResponse({"workflow_id": workflow_id})


@app.post("/api/cloud-run", response_model=CloudExecutionResponse)
async def cloud_run(request: Request, handover: HandoverPackage) -> CloudExecutionResponse:
    """Execute only the cloud-side stages for an already-redacted handover."""
    _require_mode("cloud")
    with telemetry.use_extracted_context(request.headers):
        with telemetry.tracer().start_as_current_span(
            "http.cloud_run",
            kind=trace.SpanKind.SERVER,
        ) as span:
            span.set_attribute("workflow.id", handover.workflow_id)
            span.set_attribute("http.route", "/api/cloud-run")
            state = WorkflowState(workflow_id=handover.workflow_id, handover=handover)
            state.cloud_result = await research(state)
            state.explanation = await explain(state)
    return CloudExecutionResponse(
        workflow_id=state.workflow_id,
        cloud_result=state.cloud_result,
        explanation=state.explanation,
    )


async def _sse(workflow_id: str) -> AsyncIterator[dict]:
    queue = _queues.get(workflow_id)
    if queue is None:
        yield {"event": "error", "data": json.dumps({"message": "unknown workflow"})}
        return
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            yield {"event": "done", "data": json.dumps({"workflow_id": workflow_id})}
            _queues.pop(workflow_id, None)
            return
        assert isinstance(item, StageEvent)
        yield {
            "event": f"stage.{item.stage}",
            "data": json.dumps(item.payload, ensure_ascii=False),
        }


@app.get("/api/events/{workflow_id}")
async def events(workflow_id: str) -> EventSourceResponse:
    _require_mode("edge")
    return EventSourceResponse(_sse(workflow_id))


# ---------- AG-UI protocol endpoint ----------
#
# AG-UI is an event-based protocol that streams typed events between an agent
# backend and a frontend (CopilotKit, Terminal+Agent, etc.). The protocol is
# expressive; for this demo we expose the *same* per-stage events as a single
# streaming POST so an AG-UI client can connect without the two-step
# run/events dance above.


@app.post("/agui")
async def agui(payload: dict | None = None) -> EventSourceResponse:
    _require_mode("edge")
    payload = payload or {}
    workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
    _queues[workflow_id] = asyncio.Queue()

    audio_uri = payload.get("audio_uri")
    transcript_text = payload.get("transcript")
    language_hint = payload.get("language_hint", "de-AT")
    force_violation = bool(payload.get("force_violation", False))

    asyncio.create_task(
        _drive_workflow(
            workflow_id=workflow_id,
            audio_uri=audio_uri,
            transcript_text=transcript_text,
            language_hint=language_hint,
            force_violation=force_violation,
        )
    )

    async def _agui_stream() -> AsyncIterator[dict]:
        # AG-UI-style envelope: a "run.started" event, then per-stage events,
        # then "run.finished".
        yield {
            "event": "run.started",
            "data": json.dumps({"workflow_id": workflow_id}),
        }
        async for evt in _sse(workflow_id):
            yield evt
        yield {
            "event": "run.finished",
            "data": json.dumps({"workflow_id": workflow_id}),
        }

    return EventSourceResponse(_agui_stream())


def run_server() -> None:  # entry point for the console script
    import uvicorn

    uvicorn.run(
        "hybrid_demo.ag_ui_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
