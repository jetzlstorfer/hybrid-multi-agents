from __future__ import annotations

import pytest

from hybrid_demo import config
from hybrid_demo import ag_ui_server


def test_deployment_mode_defaults_to_edge(monkeypatch):
    monkeypatch.delenv("HYBRID_DEMO_DEPLOYMENT_MODE", raising=False)
    assert config.deployment_mode() == "edge"


def test_deployment_mode_can_be_cloud(monkeypatch):
    monkeypatch.setenv("HYBRID_DEMO_DEPLOYMENT_MODE", "cloud")
    assert config.deployment_mode() == "cloud"


@pytest.mark.asyncio
async def test_status_in_cloud_mode_skips_local_models(monkeypatch):
    monkeypatch.setenv("HYBRID_DEMO_DEPLOYMENT_MODE", "cloud")
    response = await ag_ui_server.api_status()
    payload = response.body.decode("utf-8")
    assert '"mode":"cloud"' in payload
    assert '"models":[]' in payload