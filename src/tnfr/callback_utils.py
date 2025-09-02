"""Callback registration and invocation helpers."""
from __future__ import annotations

from typing import Any, Callable, DefaultDict
from enum import Enum
from collections import defaultdict
import logging

import networkx as nx

from .constants import DEFAULTS
from .trace import CallbackSpec

__all__ = ["CallbackEvent", "register_callback", "invoke_callbacks"]

logger = logging.getLogger(__name__)


class CallbackEvent(str, Enum):
    """Supported callback events."""

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"


_CALLBACK_EVENTS: tuple[str, ...] = tuple(e.value for e in CallbackEvent)


Callback = Callable[[nx.Graph, dict[str, Any]], None]
CallbackRegistry = DefaultDict[str, list[CallbackSpec]]


def _ensure_callbacks(G: nx.Graph) -> CallbackRegistry:
    """Ensure the callback structure in ``G.graph``."""
    cbs = G.graph.get("callbacks")
    if not isinstance(cbs, defaultdict):
        cbs = defaultdict(list, cbs or {})
        G.graph["callbacks"] = cbs
    return cbs


def _normalize_callback_entry(entry: Any) -> CallbackSpec | None:
    """Normalize legacy callback entries into ``CallbackSpec``.

    Parameters
    ----------
    entry:
        A previous callback entry which may be a ``CallbackSpec`` instance,
        a tuple ``(name, func)`` or simply a callable.

    Returns
    -------
    CallbackSpec | None
        Normalized ``CallbackSpec`` or ``None`` if ``entry`` cannot be
        interpreted as a callback.
    """
    if isinstance(entry, CallbackSpec):
        return entry
    if isinstance(entry, tuple):
        if not entry:
            return None
        first = entry[0]
        if isinstance(first, str):
            name = first
            fn = entry[1] if len(entry) > 1 else None
        else:
            fn = first if callable(first) else (
                entry[1] if len(entry) > 1 else None
            )
            name = getattr(fn, "__name__", None)
    else:
        fn = entry
        name = getattr(entry, "__name__", None)
    if fn is None:
        return None
    return CallbackSpec(name, fn)


def register_callback(
    G: nx.Graph,
    event: CallbackEvent | str,
    func: Callback,
    *,
    name: str | None = None,
) -> Callback:
    """Register ``func`` as callback for ``event``.

    Parameters
    ----------
    G:
        Graph where the callback registry is stored.
    event:
        Name of the event to register ``func`` under.
    func:
        Callable receiving ``(G, ctx)``.
    name:
        Optional explicit name for the callback. Defaults to ``func.__name__``.

    Returns
    -------
    Callable
        The registered ``func``.

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.callback_utils import register_callback, invoke_callbacks
    >>> G = nx.Graph()
    >>> def cb(G, ctx):
    ...     ctx.setdefault("called", 0)
    ...     ctx["called"] += 1
    >>> register_callback(G, "before_step", cb, name="counter")
    >>> ctx = {}
    >>> invoke_callbacks(G, "before_step", ctx)
    >>> ctx["called"]
    1
    """
    if event not in _CALLBACK_EVENTS:
        raise ValueError(f"Evento desconocido: {event}")
    if not callable(func):
        raise TypeError("func debe ser callable")
    cbs = _ensure_callbacks(G)

    cb_name = name or getattr(func, "__name__", None)
    new_cb = CallbackSpec(cb_name, func)

    existing_list = cbs[event]
    for i, existing in enumerate(existing_list):
        spec = _normalize_callback_entry(existing)
        if spec is None:
            continue
        if spec is not existing:
            existing_list[i] = spec
        if spec.func is func or (
            cb_name is not None and spec.name == cb_name
        ):
            existing_list[i] = new_cb
            break
    else:
        existing_list.append(new_cb)

    return func


def invoke_callbacks(
    G: nx.Graph, event: CallbackEvent | str, ctx: dict[str, Any] | None = None
) -> None:
    """Invoke all callbacks registered for ``event`` with context ``ctx``."""
    cbs = _ensure_callbacks(G).get(event, [])
    strict = bool(G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"]))
    ctx = ctx or {}
    for spec in list(cbs):
        name, fn = spec.name, spec.func
        try:
            fn(G, ctx)
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning("callback %r failed for %s: %s", name, event, e)
            if strict:
                raise
            G.graph.setdefault("_callback_errors", []).append(
                {
                    "event": event,
                    "step": ctx.get("step"),
                    "error": repr(e),
                    "fn": repr(fn),
                    "name": name,
                }
            )

