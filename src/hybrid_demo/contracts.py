"""Pydantic contracts for every payload that crosses an agent boundary.

These models replace the JSON Schema files suggested in the README. The
validation guarantee is identical and the contracts fit on a single screen
each, which matters when explaining the architecture on stage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- Transcription ----------


class TranscriptSegment(BaseModel):
    speaker: Literal["doctor", "patient", "unknown"] = "unknown"
    text: str


class Transcript(BaseModel):
    workflow_id: str
    transcript_id: str
    language: str
    segments: list[TranscriptSegment]


# ---------- PII / sensitivity ----------

EntityType = Literal[
    "PERSON_NAME",
    "DATE_OF_BIRTH",
    "AGE",
    "ADDRESS",
    "PHONE_NUMBER",
    "EMAIL",
    "INSURANCE_ID",
    "EMPLOYER",
    "LOCATION",
    "RELATIVE_NAME",
    "MEDICAL_CONDITION",
    "MEDICATION",
    "SYMPTOM",
    "PROCEDURE",
    "TIMESTAMP",
    "FREE_TEXT_IDENTIFIER",
]


class Entity(BaseModel):
    type: EntityType
    value: str
    placeholder: str | None = None


class SensitivityReport(BaseModel):
    workflow_id: str
    entities: list[Entity]


class RedactedTranscript(BaseModel):
    workflow_id: str
    redacted_transcript_id: str
    redacted_segments: list[TranscriptSegment]


# ---------- Cloud handover ----------


class PatientContext(BaseModel):
    age_bucket: str | None = None
    sex_or_gender: str | None = None
    region_type: str | None = None


class Symptom(BaseModel):
    name: str
    duration: str | None = None
    severity: str | None = None
    associated_symptoms: list[str] = Field(default_factory=list)


class HandoverPackage(BaseModel):
    workflow_id: str
    patient_context: PatientContext = Field(default_factory=PatientContext)
    chief_complaint: str | None = None
    symptoms: list[Symptom] = Field(default_factory=list)
    known_medications: list[str] = Field(default_factory=list)
    known_conditions: list[str] = Field(default_factory=list)
    negative_findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    forbidden_fields_removed: bool = True


# ---------- Cloud result ----------


class ConditionCategory(BaseModel):
    category: str
    examples: list[str] = Field(default_factory=list)
    reasoning: str
    urgency: Literal["low", "medium", "high"]


class CloudResult(BaseModel):
    workflow_id: str
    possible_condition_categories: list[ConditionCategory] = Field(
        default_factory=list)
    recommended_follow_up_questions: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Explanation(BaseModel):
    workflow_id: str
    summary: str
    clinical_reasoning: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    safety_note: str = (
        "This output is decision support only and does not replace clinical judgment."
    )


class CloudExecutionResponse(BaseModel):
    workflow_id: str
    cloud_result: CloudResult
    explanation: Explanation


# ---------- Final ----------


class FinalResponse(BaseModel):
    workflow_id: str
    patient_display_name: str | None = None
    summary_for_clinician: str
    suggested_questions: list[str] = Field(default_factory=list)
    safety_note: str = "Decision support only. Not a diagnosis."


# ---------- Policy ----------


class PolicyDecision(BaseModel):
    cloud_allowed: bool
    violations: list[str] = Field(default_factory=list)


# ---------- Internal workflow state ----------


class WorkflowState(BaseModel):
    """Mutable bag passed between executors; never serialised to the cloud."""

    workflow_id: str
    audio_uri: str | None = None
    language_hint: str | None = None
    transcript_text: str | None = None
    transcript: Transcript | None = None
    sensitivity: SensitivityReport | None = None
    redacted: RedactedTranscript | None = None
    handover: HandoverPackage | None = None
    policy: PolicyDecision | None = None
    cloud_result: CloudResult | None = None
    explanation: Explanation | None = None
    final: FinalResponse | None = None
    force_violation: bool = False
