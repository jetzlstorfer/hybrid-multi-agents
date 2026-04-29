"""Central configuration: loads ``models.yaml`` and exposes logical roles.

Agents must call :func:`get_model` with a role name like ``edge.slm`` instead
of hard-coding a model identifier. This keeps the demo's model choices in a
single auditable file (``models.yaml``) overridable per environment.

Env override pattern (Pydantic-settings nested delimiter)::

    HYBRID_DEMO__EDGE__SLM__MODEL=phi-4-mini-reasoning
    HYBRID_DEMO__CLOUD__RESEARCH__MODEL=gpt-5.4
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSpec(BaseModel):
    provider: str
    model: str
    endpoint_env: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    def endpoint(self) -> str | None:
        if self.endpoint_env:
            return os.environ.get(self.endpoint_env)
        return None


class EdgeModels(BaseModel):
    transcription: ModelSpec
    slm: ModelSpec


class CloudModels(BaseModel):
    research: ModelSpec
    explanation: ModelSpec


class ModelRegistry(BaseSettings):
    """Loaded from ``models.yaml`` and overridable via env vars."""

    edge: EdgeModels
    cloud: CloudModels

    model_config = SettingsConfigDict(
        env_prefix="HYBRID_DEMO__",
        env_nested_delimiter="__",
        extra="ignore",
    )


def _models_file() -> Path:
    return Path(os.environ.get("HYBRID_DEMO_MODELS_FILE", "models.yaml"))


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Merge HYBRID_DEMO__A__B__C=value env vars onto the YAML dict."""
    prefix = "HYBRID_DEMO__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = [p.lower() for p in key[len(prefix):].split("__")]
        node = data
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[path[-1]] = value
    return data


@lru_cache(maxsize=1)
def registry() -> ModelRegistry:
    path = _models_file()
    if not path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {path}. "
            "Set HYBRID_DEMO_MODELS_FILE or create models.yaml."
        )
    raw = yaml.safe_load(path.read_text())
    raw = _apply_env_overrides(raw)
    return ModelRegistry(**raw)


def get_model(role: str) -> ModelSpec:
    """Resolve a logical role like ``edge.slm`` or ``cloud.research``."""
    parts = role.split(".")
    obj: Any = registry()
    for part in parts:
        obj = getattr(obj, part)
    if not isinstance(obj, ModelSpec):
        raise ValueError(f"Role {role!r} did not resolve to a ModelSpec")
    return obj


def reload() -> None:
    """Drop the cached registry; useful in tests after env changes."""
    registry.cache_clear()
