"""Data boundary policy.

Single readable module. Policy lives here and nowhere else; agents must not
re-implement boundary checks. The policy fails closed: anything not explicitly
allowed is treated as forbidden.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .contracts import EntityType, HandoverPackage, PolicyDecision

# Direct identifiers that must NEVER reach the cloud.
CLOUD_FORBIDDEN_ENTITY_TYPES: frozenset[EntityType] = frozenset({
    "PERSON_NAME",
    "DATE_OF_BIRTH",
    "ADDRESS",
    "PHONE_NUMBER",
    "EMAIL",
    "INSURANCE_ID",
    "RELATIVE_NAME",
    "EMPLOYER",
    "FREE_TEXT_IDENTIFIER",
})

# Entity types that may pass through unchanged because they are clinical
# signal or non-identifying temporal context.
CLOUD_ALLOWED_CLINICAL: frozenset[EntityType] = frozenset({
    "MEDICAL_CONDITION",
    "MEDICATION",
    "SYMPTOM",
    "PROCEDURE",
    "TIMESTAMP",
})

# Top-level keys allowed inside a handover package payload sent to the cloud.
CLOUD_ALLOWED_HANDOVER_KEYS: frozenset[str] = frozenset({
    "workflow_id",
    "patient_context",
    "chief_complaint",
    "symptoms",
    "known_medications",
    "known_conditions",
    "negative_findings",
    "uncertainties",
    "forbidden_fields_removed",
})

# Forbidden patient-context keys (matches the README "Should Fail" example).
FORBIDDEN_PATIENT_CONTEXT_KEYS: frozenset[str] = frozenset({
    "name",
    "patient_name",
    "date_of_birth",
    "dob",
    "address",
    "phone",
    "email",
    "insurance_id",
})

# Final-response fields that may be rehydrated with placeholder values.
REHYDRATABLE_FIELDS: frozenset[str] = frozenset({
    "patient_display_name",
})

# Direct-identifier shaped strings that should never appear inside cloud-bound text.
_DIRECT_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Austrian SVNR: 4 digits + DDMMYY
    re.compile(r"\b\d{4}\s?\d{6}\b"),
    re.compile(r"\b\d{1,3}\s+\w+(?:gasse|straße|strasse|platz|weg)\b", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b\+?\d[\d\s\-/()]{7,}\b"),
)


# ---------- Validation ----------


def _scan_for_direct_identifier_strings(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in _DIRECT_IDENTIFIER_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def validate_handover(payload: HandoverPackage | dict[str, Any]) -> PolicyDecision:
    """Validate a cloud-bound handover package. Fails closed.

    Accepts either a typed :class:`HandoverPackage` or a raw dict (used in tests
    that mirror the README's "Should Fail" payload).
    """
    if isinstance(payload, HandoverPackage):
        data = payload.model_dump()
    else:
        data = dict(payload)

    violations: list[str] = []

    # Top-level key allowlist.
    for key in data.keys():
        if key not in CLOUD_ALLOWED_HANDOVER_KEYS:
            violations.append(f"Disallowed top-level field: {key}")

    # Patient-context: only allowlisted keys; explicit forbidden-key messages.
    ctx = data.get("patient_context") or {}
    if isinstance(ctx, dict):
        for key in ctx.keys():
            if key in FORBIDDEN_PATIENT_CONTEXT_KEYS:
                violations.append(f"Direct identifier present: patient_context.{key}")

    # Scan all string leaves for identifier-shaped values.
    serialised = json.dumps(data, ensure_ascii=False)
    for hit in _scan_for_direct_identifier_strings(serialised):
        violations.append(f"Identifier-shaped string detected (pattern: {hit})")

    return PolicyDecision(cloud_allowed=not violations, violations=violations)
