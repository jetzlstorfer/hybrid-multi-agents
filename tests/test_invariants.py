"""Workflow-level invariants.

The pipeline is intentionally fail-fast and no longer provides fallback
execution paths. These tests verify that missing required inputs fail early
and that the cloud-boundary guard in the vault still holds.
"""

from __future__ import annotations

import pytest

from hybrid_demo import vault
from hybrid_demo.workflow import run_workflow


@pytest.mark.asyncio
async def test_workflow_requires_audio_uri():
    with pytest.raises(ValueError, match="audio_uri is required"):
        async for _ in run_workflow():
            pass


@pytest.mark.asyncio
async def test_force_violation_still_requires_audio_uri():
    with pytest.raises(ValueError, match="audio_uri is required"):
        async for _ in run_workflow(force_violation=True):
            pass


def test_vault_blocks_cloud_context():
    token = vault.cloud_context()
    try:
        with pytest.raises(vault.CloudBoundaryViolation):
            vault.reveal("wf_test", "[PATIENT_FIRST_NAME]")
    finally:
        vault.reset_cloud_context(token)
