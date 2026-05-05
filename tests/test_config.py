"""Configuration tests: env override flips the resolved model name."""

from __future__ import annotations

import pytest

from hybrid_demo import config


def test_default_slm_is_configured(monkeypatch):
    monkeypatch.delenv("HYBRID_DEMO__EDGE__SLM__MODEL", raising=False)
    config.reload()
    # The default SLM is whatever is declared in models.yaml; just check the
    # spec resolves to a non-empty model name rather than a specific value so
    # the test doesn't need updating every time models.yaml changes.
    assert config.get_model("edge.slm").model


def test_env_override_changes_model(monkeypatch):
    monkeypatch.setenv("HYBRID_DEMO__EDGE__SLM__MODEL", "phi-4-mini-reasoning")
    config.reload()
    assert config.get_model("edge.slm").model == "phi-4-mini-reasoning"


def test_unknown_role_raises():
    config.reload()
    with pytest.raises(AttributeError):
        config.get_model("edge.does_not_exist")
