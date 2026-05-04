"""Process-local placeholder vault.

Maps placeholder tokens (e.g. ``[PATIENT_FIRST_NAME]``) back to the original values.
The vault is in-memory only, never persisted, never serialised to anything
that could be sent to the cloud. Cloud-tagged code paths must not call
:func:`reveal`.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Final

_lock = threading.Lock()
_store: dict[str, dict[str, str]] = {}

# Set inside cloud-side executors. Any reveal() call while this is True is a
# defensive guard failure (programmer error).
_in_cloud_context: Final[contextvars.ContextVar[bool]] = contextvars.ContextVar(
    "hybrid_demo_in_cloud_context", default=False
)


class CloudBoundaryViolation(RuntimeError):
    """Raised when vault access happens inside a cloud-tagged context."""


def cloud_context() -> contextvars.Token[bool]:
    """Mark the current async/sync context as cloud-side."""
    return _in_cloud_context.set(True)


def reset_cloud_context(token: contextvars.Token[bool]) -> None:
    _in_cloud_context.reset(token)


def store(workflow_id: str, mapping: dict[str, str]) -> None:
    with _lock:
        _store.setdefault(workflow_id, {}).update(mapping)


def reveal(workflow_id: str, placeholder: str) -> str | None:
    if _in_cloud_context.get():
        raise CloudBoundaryViolation(
            "vault.reveal() called from a cloud-tagged context"
        )
    with _lock:
        return _store.get(workflow_id, {}).get(placeholder)


def forget(workflow_id: str) -> None:
    with _lock:
        _store.pop(workflow_id, None)
