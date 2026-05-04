"""Configuration tests: env override flips the resolved model name."""

from __future__ import annotations

import pytest

from hybrid_demo import config


def test_default_slm_is_phi4(monkeypatch):
    monkeypatch.delenv("HYBRID_DEMO__EDGE__SLM__MODEL", raising=False)
    config.reload()
    assert config.get_model("edge.slm").model == "phi-4-mini"


def test_env_override_changes_model(monkeypatch):
    monkeypatch.setenv("HYBRID_DEMO__EDGE__SLM__MODEL", "phi-4-mini-reasoning")
    config.reload()
    assert config.get_model("edge.slm").model == "phi-4-mini-reasoning"


def test_unknown_role_raises():
    config.reload()
    with pytest.raises(AttributeError):
        config.get_model("edge.does_not_exist")
