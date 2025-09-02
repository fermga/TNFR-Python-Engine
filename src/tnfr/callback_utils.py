"""Callback registration and invocation helpers."""
from __future__ import annotations

from typing import Callable, Any, Optional
from enum import Enum
from collections import defaultdict
import logging

from .constants import DEFAULTS

__all__ = ["CallbackEvent", "register_callback", "invoke_callbacks"]

logger = logging.getLogger(__name__)


class CallbackEvent(str, Enum):
    """Supported callback events."""

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"


_CALLBACK_EVENTS = tuple(e.value for e in CallbackEvent)


def _ensure_callbacks(G):
    """Ensure the callback structure in ``G.graph``."""
    cbs = G.graph.get("callbacks")
    if not isinstance(cbs, defaultdict):
        cbs = defaultdict(list, cbs or {})
        G.graph["callbacks"] = cbs
    return cbs


def register_callback(
    G,
    event: CallbackEvent | str,
    func: Optional[Callable] = None,
    *,
    name: str | None = None,
):
    """Register ``func`` as callback for ``event``."""
    if event not in _CALLBACK_EVENTS:
        raise ValueError(f"Evento desconocido: {event}")
    if func is None:
        raise TypeError("func es obligatorio")
    cbs = _ensure_callbacks(G)

    if isinstance(func, tuple):
        cb_name, func = func
    else:
        cb_name = name or getattr(func, "__name__", None)

    new_cb = (cb_name, func)

    for i, (existing_name, existing_fn) in enumerate(cbs[event]):
        if existing_fn is func or (cb_name is not None and existing_name == cb_name):
            cbs[event][i] = new_cb
            break
    else:
        cbs[event].append(new_cb)

    return func


def invoke_callbacks(G, event: CallbackEvent | str, ctx: dict | None = None):
    """Invoke all callbacks registered for ``event`` with context ``ctx``."""
    cbs = _ensure_callbacks(G).get(event, [])
    strict = bool(G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"]))
    ctx = ctx or {}
    for name, fn in list(cbs):
        try:
            fn(G, ctx)
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning("callback %r failed for %s: %s", name, event, e)
            if strict:
                raise
            G.graph.setdefault("_callback_errors", []).append({
                "event": event,
                "step": ctx.get("step"),
                "error": repr(e),
                "fn": repr(fn),
                "name": name,
            })

