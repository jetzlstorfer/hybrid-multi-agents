"""Edge rehydration agent.

Restores selected placeholders in the final, clinician-facing response.
Deterministic. Reads from the local vault. Only fields whitelisted in
``policy.REHYDRATABLE_FIELDS`` are populated; everything else is left as the
cloud agents produced it.
"""

from __future__ import annotations

from .. import vault
from ..contracts import Explanation, FinalResponse, WorkflowState
from ..policy import REHYDRATABLE_FIELDS
from ..telemetry import traced_step


@traced_step(
    name="edge.rehydration",
    runtime_location="edge",
    input_classification="cloud_explanation",
    output_classification="final_response",
)
def rehydrate(state: WorkflowState) -> FinalResponse:
    explanation: Explanation = state.explanation  # type: ignore[assignment]

    final = FinalResponse(
        workflow_id=state.workflow_id,
        summary_for_clinician=explanation.summary,
        suggested_questions=(
            state.cloud_result.recommended_follow_up_questions
            if state.cloud_result
            else []
        ),
    )

    if "patient_display_name" in REHYDRATABLE_FIELDS:
        first = vault.reveal(state.workflow_id, "[PATIENT_FIRST_NAME]")
        last = vault.reveal(state.workflow_id, "[PATIENT_LAST_NAME]")
        parts = [p for p in [first, last] if p]
        final.patient_display_name = " ".join(parts) if parts else None

    return final
