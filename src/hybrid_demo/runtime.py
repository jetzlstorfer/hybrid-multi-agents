"""Lazy singletons for the local Foundry runtime and the cloud chat client.

Module split so test code can stub these without importing the heavy SDKs.
"""

from __future__ import annotations

import logging
import os
import re
from threading import Lock
from typing import Any

from . import config

_log = logging.getLogger(__name__)

_local_lock = Lock()
_cloud_lock = Lock()
_local_audio_client: Any = None
_local_chat_client: Any = None
_cloud_chat_client: Any = None

# Cache keyed by role for openai-compatible singletons (thread-safe enough for
# our single-worker demo; extend with per-key locks if needed).
_openai_edge_clients: dict[str, Any] = {}
_openai_cloud_clients: dict[str, Any] = {}


# ---------- OpenAI-compatible edge wrapper ----------


class _OpenAICompatibleEdgeClient:
    """Thin adapter that wraps an ``openai.OpenAI`` client and exposes the
    ``complete_chat(messages, tools=None)`` interface expected by edge agents.

    The returned response objects are standard ``openai.ChatCompletion`` objects
    so ``.choices[0].message.content`` works unchanged.

    Reasoning models (e.g. Qwen3, DeepSeek-R1) wrap their chain-of-thought in
    ``<think>...</think>`` tags before the final answer.  This adapter strips
    those blocks so downstream JSON parsing is not affected.
    """

    #: Signals to callers (e.g. pii_agent) that this client supports
    #: JSON mode via ``response_format={"type": "json_object"}``.
    json_mode_supported: bool = True

    def __init__(self, openai_client: Any, model: str, options: dict) -> None:
        self._client = openai_client
        self._model = model
        self._options = options

    def complete_chat(
        self,
        messages: list[dict],
        tools: list | None = None,
        response_format: dict | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        # Separate standard OpenAI params from llama.cpp/vLLM-specific extensions.
        # ``enable_thinking`` must travel via ``extra_body`` because the OpenAI
        # Python client rejects unknown top-level kwargs.
        _EXTRA_BODY_OPTIONS = {"enable_thinking"}
        extra_body: dict[str, Any] = {}
        for key, value in self._options.items():
            if key in _EXTRA_BODY_OPTIONS:
                extra_body[key] = value
            elif key == "extra_body" and isinstance(value, dict):
                # Already-nested extra_body dict from models.yaml — merge in.
                extra_body.update(value)
            else:
                kwargs[key] = value
        if tools:
            kwargs["tools"] = tools

        # When the server is running the model in thinking mode the JSON grammar
        # constraint (response_format) must NOT be sent: most inference servers
        # apply the grammar from the very first token, which conflicts with the
        # leading <think>…</think> block and produces empty or garbled output.
        # Suppress response_format for the entire request; the system prompts
        # already instruct the model to return JSON as its final answer.
        _thinking_on = extra_body.get("enable_thinking", False)
        if response_format is not None and not _thinking_on:
            kwargs["response_format"] = response_format

        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Some OpenAI-compatible gateways (especially behind ingress/proxy)
            # return HTTP 504 for long reasoning generations. Retry once with a
            # tighter completion token budget while keeping reasoning enabled.
            if _is_gateway_timeout_error(exc):
                retry_kwargs = _tighten_token_budget(kwargs)
                if retry_kwargs is not kwargs:
                    _log.warning(
                        "OpenAI-compatible request timed out (504). "
                        "Retrying with tighter token budget."
                    )
                    response = self._client.chat.completions.create(**retry_kwargs)
                else:
                    raise
            else:
                raise

        # Strip thinking tokens so callers always receive the final answer text.
        for choice in response.choices:
            if choice.message and choice.message.content:
                raw = choice.message.content
                stripped = _strip_thinking_tags(raw)
                _log.debug(
                    "raw response (%d chars, finish_reason=%s): %.500s",
                    len(raw),
                    choice.finish_reason,
                    raw,
                )
                if not stripped:
                    _log.warning(
                        "Response content became empty after stripping thinking tags "
                        "(finish_reason=%s, raw_length=%d). "
                        "The model likely exhausted max_tokens inside <think> with no answer remaining. "
                        "Increase max_tokens (e.g. 8192) or add budget_tokens to extra_body.",
                        choice.finish_reason,
                        len(raw),
                    )
                choice.message.content = stripped
        return response


_THINKING_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Matches an opening <think> tag with no matching closing tag — happens when
# the token budget is exhausted mid-reasoning so the model never writes </think>.
_THINKING_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_thinking_tags(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by reasoning models.

    Also strips unclosed ``<think>`` blocks caused by token-budget exhaustion
    so downstream JSON parsing never receives raw reasoning tokens.
    """
    result = _THINKING_TAG_RE.sub("", text)
    result = _THINKING_OPEN_RE.sub("", result)
    return result.strip()


def _is_gateway_timeout_error(exc: Exception) -> bool:
    """Return True when an OpenAI-compatible request failed with HTTP 504."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 504:
        return True

    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 504:
        return True

    text = str(exc).lower()
    return "504" in text or "gateway time-out" in text or "gateway timeout" in text


def _tighten_token_budget(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a retry kwargs dict with lower completion token budget.

    Keeps reasoning enabled, but caps generation length to reduce proxy timeout
    risk on long-running reasoning responses.
    """
    out = dict(kwargs)

    # OpenAI-compatible APIs may use either key; handle both conservatively.
    max_tokens = out.get("max_tokens")
    max_completion_tokens = out.get("max_completion_tokens")

    updated = False
    if isinstance(max_tokens, int) and max_tokens > 1024:
        out["max_tokens"] = 1024
        updated = True
    if isinstance(max_completion_tokens, int) and max_completion_tokens > 1024:
        out["max_completion_tokens"] = 1024
        updated = True

    return out if updated else kwargs


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
    """Return a chat client for the configured edge SLM.

    When ``provider`` is ``openai-compatible`` the client is a thin
    :class:`_OpenAICompatibleEdgeClient` wrapper backed by the ``openai``
    library; otherwise a Foundry Local SDK chat client is returned.
    """
    global _local_chat_client
    spec = config.get_model("edge.slm")

    if spec.provider == "openai-compatible":
        role = "edge.slm"
        if role in _openai_edge_clients:
            return _openai_edge_clients[role]
        with _local_lock:
            if role in _openai_edge_clients:
                return _openai_edge_clients[role]
            client = _make_openai_edge_client(spec)
            _openai_edge_clients[role] = client
            _log.info("Created openai-compatible edge client: model=%s base_url=%s", spec.model, spec.endpoint())
        return _openai_edge_clients[role]

    if _local_chat_client is not None:
        return _local_chat_client
    with _local_lock:
        if _local_chat_client is not None:
            return _local_chat_client
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


# ---------- OpenAI-compatible factories ----------


def _make_openai_edge_client(spec: "config.ModelSpec") -> _OpenAICompatibleEdgeClient:
    """Create an :class:`_OpenAICompatibleEdgeClient` from a ModelSpec."""
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for provider 'openai-compatible'. "
            "Install with `pip install openai`."
        ) from exc

    endpoint = spec.endpoint()
    if not endpoint:
        raise RuntimeError(
            "provider 'openai-compatible' requires 'base_url' or 'endpoint_env' in models.yaml."
        )
    api_key = spec.api_key() or "no-key"  # some local servers don't verify the key
    # Use a generous HTTP timeout so long reasoning generations on CPU-based
    # inference servers (llama.cpp) are not aborted client-side before the
    # proxy/Route timeout fires.  The default openai client timeout is 600s
    # for read but only 5s for connect; we set both explicitly.
    raw_client = openai.OpenAI(
        base_url=endpoint,
        api_key=api_key,
        timeout=openai.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
    )
    return _OpenAICompatibleEdgeClient(raw_client, spec.model, spec.options)


def _make_openai_cloud_client(spec: "config.ModelSpec") -> Any:
    """Create an ``OpenAIChatCompletionClient`` for a cloud role."""
    try:
        from agent_framework.openai import OpenAIChatCompletionClient
    except ImportError as exc:
        raise RuntimeError(
            "agent-framework-openai is required for provider 'openai-compatible' in cloud roles. "
            "Install with `pip install agent-framework-openai`."
        ) from exc

    endpoint = spec.endpoint()
    if not endpoint:
        raise RuntimeError(
            "provider 'openai-compatible' requires 'base_url' or 'endpoint_env' in models.yaml."
        )
    api_key = spec.api_key() or "no-key"
    return OpenAIChatCompletionClient(model=spec.model, base_url=endpoint, api_key=api_key)


# ---------- Cloud (Microsoft Foundry / OpenAI-compatible) ----------


def get_cloud_chat_client(role: str = "cloud.research"):
    """Return a chat client for the given cloud role.

    When ``provider`` is ``openai-compatible`` an ``OpenAIChatCompletionClient``
    is returned (no Azure credential required).  Otherwise a ``FoundryChatClient``
    backed by Azure identity is returned.
    """
    spec = config.get_model(role)

    if spec.provider == "openai-compatible":
        if role in _openai_cloud_clients:
            return _openai_cloud_clients[role]
        with _cloud_lock:
            if role in _openai_cloud_clients:
                return _openai_cloud_clients[role]
            client = _make_openai_cloud_client(spec)
            _openai_cloud_clients[role] = client
            _log.info("Created openai-compatible cloud client: role=%s model=%s", role, spec.model)
        return _openai_cloud_clients[role]

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
