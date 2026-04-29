"""Lazy singletons for the local Foundry runtime and the cloud chat client.

Module split so test code can stub these without importing the heavy SDKs.
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

from . import config

_log = logging.getLogger(__name__)

_local_lock = Lock()
_cloud_lock = Lock()
_local_audio_client: Any = None
_local_chat_client: Any = None
_cloud_chat_client: Any = None


def _close_if_supported(obj: Any) -> None:
    """Best-effort close for SDK clients that expose cleanup hooks."""
    if obj is None:
        return
    for method_name in ("close", "shutdown", "dispose"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:  # pragma: no cover
                pass
            return


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------- Local (Foundry Local) ----------


def _ensure_foundry_local() -> Any:
    """Return the FoundryLocalManager instance, initialising it lazily."""
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except ImportError as exc:  # pragma: no cover - depends on private feed
        raise RuntimeError(
            "foundry-local-sdk is not installed. Install with `pip install -e .[local]`."
        ) from exc

    if FoundryLocalManager.instance is None:
        cfg = Configuration(app_name="hybrid_demo")
        FoundryLocalManager.initialize(cfg)
        manager = FoundryLocalManager.instance
        manager.download_and_register_eps()
    return FoundryLocalManager.instance


def get_local_audio_client():
    """Return a Foundry Local audio client for the configured Whisper model."""
    global _local_audio_client
    if _local_audio_client is not None:
        return _local_audio_client
    with _local_lock:
        if _local_audio_client is not None:
            return _local_audio_client
        spec = config.get_model("edge.transcription")
        manager = _ensure_foundry_local()
        model = manager.catalog.get_model(spec.model)
        model.download()
        model.load()
        _local_audio_client = model.get_audio_client()
        _log.info("Loaded local audio model: %s", spec.model)
    return _local_audio_client


def get_local_chat_client():
    """Return a Foundry Local chat client for the configured SLM."""
    global _local_chat_client
    if _local_chat_client is not None:
        return _local_chat_client
    with _local_lock:
        if _local_chat_client is not None:
            return _local_chat_client
        spec = config.get_model("edge.slm")
        manager = _ensure_foundry_local()
        model = manager.catalog.get_model(spec.model)
        model.download()
        model.load()
        _local_chat_client = model.get_chat_client()

        # Apply model options from models.yaml so reasoning models stay bounded.
        settings = getattr(_local_chat_client, "settings", None)
        if settings is not None:
            max_tokens = _as_int(spec.options.get("max_tokens"))
            temperature = _as_float(spec.options.get("temperature"))
            top_p = _as_float(spec.options.get("top_p"))
            top_k = _as_int(spec.options.get("top_k"))

            if max_tokens is not None and hasattr(settings, "max_tokens"):
                settings.max_tokens = max_tokens
            if temperature is not None and hasattr(settings, "temperature"):
                settings.temperature = temperature
            if top_p is not None and hasattr(settings, "top_p"):
                settings.top_p = top_p
            if top_k is not None and hasattr(settings, "top_k"):
                settings.top_k = top_k

            _log.info(
                "Applied local SLM settings: max_tokens=%s temperature=%s top_p=%s top_k=%s",
                getattr(settings, "max_tokens", None),
                getattr(settings, "temperature", None),
                getattr(settings, "top_p", None),
                getattr(settings, "top_k", None),
            )

        _log.info("Loaded local SLM: %s", spec.model)
    return _local_chat_client


# ---------- Cloud (Microsoft Foundry) ----------


def get_cloud_chat_client(role: str = "cloud.research"):
    """Return a FoundryChatClient for the given cloud role.

    Roles share a project endpoint but may use different models.
    """
    spec = config.get_model(role)
    endpoint = spec.endpoint() or os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            f"Role {role!r} requires {spec.endpoint_env or 'FOUNDRY_PROJECT_ENDPOINT'}."
        )

    from agent_framework.foundry import FoundryChatClient
    from azure.identity import AzureCliCredential

    return FoundryChatClient(
        project_endpoint=endpoint,
        model=spec.model,
        credential=AzureCliCredential(),
    )


def unload() -> None:
    """Release loaded local models. Safe to call multiple times."""
    global _local_audio_client, _local_chat_client
    _close_if_supported(_local_audio_client)
    _close_if_supported(_local_chat_client)

    try:
        from foundry_local_sdk import FoundryLocalManager

        if FoundryLocalManager.instance is not None:
            for model in FoundryLocalManager.instance.catalog.get_loaded_models():
                try:
                    model.unload()
                except Exception:  # pragma: no cover
                    pass
            try:
                FoundryLocalManager.instance.stop_web_service()
            except Exception:  # pragma: no cover
                pass
    except ImportError:
        pass
    _local_audio_client = None
    _local_chat_client = None
