"""Callback registration and invocation helpers."""

from __future__ import annotations


from typing import Any, TYPE_CHECKING
from enum import Enum
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence

import traceback
from .logging_utils import get_logger

from .constants import DEFAULTS
from .trace import CallbackSpec

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

__all__ = ["CallbackEvent", "register_callback", "invoke_callbacks"]

logger = get_logger(__name__)


class CallbackEvent(str, Enum):
    """Supported callback events."""

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"


_CALLBACK_EVENTS: set[str] = {e.value for e in CallbackEvent}

_CALLBACK_ERROR_LIMIT = 100  # keep only this many recent callback errors

Callback = Callable[["nx.Graph", dict[str, Any]], None]
CallbackRegistry = dict[str, list["CallbackSpec"]]


def _ensure_callbacks(G: "nx.Graph") -> CallbackRegistry:
    """Ensure the callback structure in ``G.graph``."""
    cbs = G.graph.get("callbacks")

    # Defensive: if callbacks store is not a mapping, discard it to avoid
    # failures when constructing the defaultdict below.
    if not isinstance(cbs, Mapping):
        logger.warning(
            "Invalid callbacks registry on graph; resetting to empty"
        )
        cbs = defaultdict(list)
        G.graph["callbacks"] = cbs
        G.graph["_callbacks_dirty"] = True
    elif not isinstance(cbs, defaultdict):
        cbs = defaultdict(list, cbs)
        G.graph["callbacks"] = cbs
        G.graph["_callbacks_dirty"] = True
    if not G.graph.pop("_callbacks_dirty", False):
        return cbs
    for event in list(cbs):
        if event not in _CALLBACK_EVENTS:
            del cbs[event]
            continue
        lst = cbs[event]
        cbs[event] = [
            spec
            for entry in lst
            if (spec := _normalize_callback_entry(entry)) is not None
        ]
    G.graph["_callbacks_dirty"] = False
    return cbs


def _normalize_event(event: CallbackEvent | str) -> str:
    """Return ``event`` as a string."""
    return event.value if isinstance(event, CallbackEvent) else str(event)


def _normalize_callback_entry(entry: Any) -> "CallbackSpec | None":
    """Normalize a callback specification.

    Supported formats
    -----------------
    * :class:`CallbackSpec` instances (returned unchanged).
    * Sequences ``(name: str, func: Callable)`` such as lists or tuples.
    * Bare callables ``func`` whose name is taken from ``func.__name__``.

    ``None`` is returned when ``entry`` does not match any of the accepted
    formats.  The original ``entry`` is never mutated.
    """

    if isinstance(entry, CallbackSpec):
        return entry
    elif isinstance(entry, Sequence) and not isinstance(entry, str):
        if len(entry) != 2:
            return None
        name, fn = entry
        if not isinstance(name, str) or not callable(fn):
            return None
        return CallbackSpec(name, fn)
    elif callable(entry):
        name = getattr(entry, "__name__", None)
        return CallbackSpec(name, entry)
    else:
        return None


def register_callback(
    G: "nx.Graph",
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
    event = _normalize_event(event)
    if event not in _CALLBACK_EVENTS:
        raise ValueError(f"Unknown event: {event}")
    if not callable(func):
        raise TypeError("func must be callable")
    cbs = _ensure_callbacks(G)

    cb_name = name or getattr(func, "__name__", None)
    new_cb = CallbackSpec(cb_name, func)

    existing_list = cbs[event]
    for i, spec in enumerate(existing_list):
        if spec.func is func or (cb_name is not None and spec.name == cb_name):
            existing_list[i] = new_cb
            break
    else:
        existing_list.append(new_cb)
    G.graph["_callbacks_dirty"] = True
    return func


def invoke_callbacks(
    G: "nx.Graph",
    event: CallbackEvent | str,
    ctx: dict[str, Any] | None = None,
) -> None:
    """Invoke all callbacks registered for ``event`` with context ``ctx``."""
    event = _normalize_event(event)
    cbs = _ensure_callbacks(G).get(event, [])
    strict = bool(
        G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"])
    )
    ctx = ctx or {}
    err_list = G.graph.get("_callback_errors")
    if not isinstance(err_list, deque):
        err_list = deque(maxlen=_CALLBACK_ERROR_LIMIT)
        G.graph["_callback_errors"] = err_list
    # ``cbs`` is a list and callbacks are not modified during iteration,
    # so iterating directly avoids an unnecessary copy.
    for spec in cbs:
        name, fn = spec.name, spec.func
        try:
            fn(G, ctx)
        except Exception as e:  # catch all callback errors
            logger.exception("callback %r failed for %s: %s", name, event, e)
            if strict:
                raise
            err_list.append(
                {
                    "event": event,
                    "step": ctx.get("step"),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "fn": repr(fn),
                    "name": name,
                }
            )
