from __future__ import annotations

import threading
import traceback
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import Any, TypedDict

import networkx as nx

from .constants import DEFAULTS
from .locking import get_lock
from .trace import CallbackSpec
from .utils import get_logger, is_non_string_sequence

__all__ = (
    "CallbackEvent",
    "CallbackManager",
    "callback_manager",
    "CallbackError",
)

logger: Any


class CallbackEvent(str, Enum):
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"


Callback = Callable[[nx.Graph, dict[str, Any]], None]
CallbackRegistry = dict[str, dict[str, CallbackSpec]]


class CallbackError(TypedDict):
    event: str
    step: int | None
    error: str
    traceback: str
    fn: str
    name: str | None


class CallbackManager:
    def __init__(self) -> None: ...

    def get_callback_error_limit(self) -> int: ...

    def set_callback_error_limit(self, limit: int) -> int: ...

    def register_callback(
        self,
        G: nx.Graph,
        event: CallbackEvent | str,
        func: Callback,
        *,
        name: str | None = ...,
    ) -> Callback: ...

    def invoke_callbacks(
        self,
        G: nx.Graph,
        event: CallbackEvent | str,
        ctx: dict[str, Any] | None = ...,
    ) -> None: ...

    def _record_callback_error(
        self,
        G: nx.Graph,
        event: str,
        ctx: dict[str, Any],
        spec: CallbackSpec,
        err: Exception,
    ) -> None: ...

    def _ensure_callbacks_nolock(self, G: nx.Graph) -> CallbackRegistry: ...

    def _ensure_callbacks(self, G: nx.Graph) -> CallbackRegistry: ...


callback_manager: CallbackManager


def _func_id(fn: Callable[..., Any]) -> str: ...

def _validate_registry(G: nx.Graph, cbs: Any, dirty: set[str]) -> CallbackRegistry: ...

def _normalize_callbacks(entries: Any) -> dict[str, CallbackSpec]: ...

def _normalize_event(event: CallbackEvent | str) -> str: ...

def _is_known_event(event: str) -> bool: ...

def _ensure_known_event(event: str) -> None: ...

def _normalize_callback_entry(entry: Any) -> CallbackSpec | None: ...

def _reconcile_callback(
    event: str,
    existing_map: dict[str, CallbackSpec],
    spec: CallbackSpec,
    strict: bool,
) -> str: ...
