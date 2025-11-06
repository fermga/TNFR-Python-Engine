from __future__ import annotations

import networkx as nx
from ..types import CallbackError as CallbackError
from _typeshed import Incomplete
from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple

__all__ = [
    "CallbackEvent",
    "CallbackManager",
    "callback_manager",
    "CallbackError",
    "CallbackSpec",
]

Callback = Callable[[nx.Graph, dict[str, Any]], None]

class CallbackSpec(NamedTuple):
    name: str | None
    func: Callable[..., Any]

class CallbackEvent(str, Enum):
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"
    CACHE_METRICS = "cache_metrics"

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
        name: str | None = None,
    ) -> Callback: ...
    def invoke_callbacks(
        self, G: nx.Graph, event: CallbackEvent | str, ctx: dict[str, Any] | None = None
    ) -> None: ...

callback_manager: Incomplete

def _normalize_callbacks(entries: Any) -> dict[str, CallbackSpec]: ...
def _normalize_callback_entry(entry: Any) -> CallbackSpec | None: ...
