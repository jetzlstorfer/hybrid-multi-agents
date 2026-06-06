"""Live integration tests for SLM-driven redaction.

These tests exercise the configured local SLM redaction path end-to-end.
They are opt-in to keep default CI fast and deterministic.

Enable with:

    RUN_SLM_INTEGRATION=1 pytest -q tests/test_redaction_integration.py
"""

from __future__ import annotations

import os
import gc

import pytest

from hybrid_demo.contracts import (
    Entity,
    SensitivityReport,
    Transcript,
    TranscriptSegment,
    WorkflowState,
)
from hybrid_demo.edge.redaction_agent import redact


def _integration_enabled() -> bool:
    return os.environ.get("RUN_SLM_INTEGRATION") == "1"


def _state(text: str, entities: list[Entity]) -> WorkflowState:
    state = WorkflowState(workflow_id="wf_integration")
    state.transcript = Transcript(
        workflow_id="wf_integration",
        transcript_id="tr_integration",
        language="de-AT",
        segments=[TranscriptSegment(speaker="unknown", text=text)],
    )
    state.sensitivity = SensitivityReport(
        workflow_id="wf_integration",
        entities=entities,
    )
    return state


@pytest.fixture(scope="module", autouse=True)
def require_live_slm():
    from hybrid_demo import runtime

    if not _integration_enabled():
        pytest.skip(
            "Set RUN_SLM_INTEGRATION=1 to run live SLM integration tests")

    # Fail fast when explicitly requested but unavailable/misconfigured.
    try:
        runtime.get_local_chat_client()
    except Exception as exc:  # pragma: no cover
        pytest.fail(
            f"RUN_SLM_INTEGRATION=1 but local SLM is unavailable: {exc}")

    yield

    # Integration tests should leave no live local-runtime objects behind.
    runtime.unload()
    gc.collect()


def test_live_slm_redacts_repeated_names_and_email():
    text = (
        "Darf ich erfahren, wie Sie heißen? Paul Gerster. "
        "Herr Gerster, ich notiere Ihre E-Mail paul.gerster@example.com."
    )
    entities = [
        Entity(
            type="PERSON_NAME",
            value="Paul Gerster",
            placeholder="[PATIENT_FIRST_NAME] [PATIENT_LAST_NAME]",
        ),
        Entity(type="EMAIL", value="paul.gerster@example.com",
               placeholder="[EMAIL]"),
    ]

    import asyncio
    out = asyncio.run(redact(_state(text, entities))).redacted_segments[0].text

    assert "Paul Gerster" not in out
    assert "Herr Gerster" not in out
    assert "paul.gerster@example.com" not in out
    assert "[PATIENT_FIRST_NAME]" in out
    assert "[PATIENT_LAST_NAME]" in out
    assert "[EMAIL]" in out


def test_live_slm_applies_configured_transform_actions():
    text = "Ich bin 42 Jahre alt und wohne in Wien."
    entities = [
        Entity(type="AGE", value="42", placeholder="[PATIENT_AGE]"),
        Entity(type="LOCATION", value="Wien", placeholder="[LOCATION]"),
    ]

    import asyncio
    out = asyncio.run(redact(_state(text, entities))).redacted_segments[0].text
    lowered = out.casefold()

    # We expect either an actual generalisation or an explicit placeholder
    # replacement, but never an untouched source string for both fields.
    age_changed = ("42" not in out) or (
        "[PATIENT_AGE]" in out) or ("age bucket" in lowered)
    location_changed = (
        ("wien" not in lowered)
        or ("[LOCATION]" in out)
        or ("urban" in lowered)
        or ("region" in lowered)
    )

    assert age_changed
    assert location_changed
