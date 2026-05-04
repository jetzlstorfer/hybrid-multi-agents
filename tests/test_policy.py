"""Policy-gate tests.

The README defines two example payloads. The "Should Pass" payload must be
admitted; the "Should Fail" payload must be rejected with violations naming
the three direct identifiers.
"""

from __future__ import annotations

from hybrid_demo.policy import validate_handover


def test_readme_should_pass():
    payload = {
        "patient_context": {
            "age_bucket": "40-49",
            "region_type": "urban area",
        },
        "chief_complaint": "Chest pain for 3 days",
        "symptoms": ["chest pain"],
        "uncertainties": ["No vital signs provided"],
    }
    decision = validate_handover(payload)
    assert decision.cloud_allowed is True
    assert decision.violations == []


def test_readme_should_fail():
    payload = {
        "patient_context": {
            "name": "Anna Müller",
            "date_of_birth": "1983-04-18",
            "address": "Example Street 12, Vienna",
        },
        "chief_complaint": "Chest pain for 3 days",
    }
    decision = validate_handover(payload)
    assert decision.cloud_allowed is False
    msgs = decision.violations
    assert "Direct identifier present: patient_context.name" in msgs
    assert "Direct identifier present: patient_context.date_of_birth" in msgs
    assert "Direct identifier present: patient_context.address" in msgs


def test_disallowed_top_level_field():
    payload = {
        "patient_context": {"age_bucket": "40-49"},
        "raw_transcript": "Anna Müller, 42, aus Wien.",
    }
    decision = validate_handover(payload)
    assert decision.cloud_allowed is False
    assert any("raw_transcript" in v for v in decision.violations)
