"""Callback registration and invocation helpers.

This module is thread-safe: all mutations of the callback registry stored in a
graph's ``G.graph`` are serialised using a process-wide lock obtained via
``locking.get_lock("callbacks")``. Callback functions themselves execute
outside of the lock and must therefore be independently thread-safe if they
modify shared state.
"""

from __future__ import annotations


from typing import Any, TypedDict
from enum import Enum
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Iterable

import traceback
import threading
from .logging import get_module_logger
from .constants import DEFAULTS
from .locking import get_lock

from .trace import CallbackSpec
from .collections_utils import is_non_string_sequence

import networkx as nx  # type: ignore[import-untyped]

__all__ = (
    "CallbackEvent",
    "register_callback",
    "invoke_callbacks",
    "get_callback_error_limit",
    "set_callback_error_limit",
    "CallbackError",
)

logger = get_module_logger(__name__)

_CALLBACK_LOCK = get_lock("callbacks")


class CallbackEvent(str, Enum):
    """Supported callback events."""

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"


_CALLBACK_EVENTS: set[str] = {e.value for e in CallbackEvent}

# Default number of recent callback errors to retain.
# Use ``set_callback_error_limit`` to adjust.
_CALLBACK_ERROR_LIMIT_LOCK = threading.Lock()
_CALLBACK_ERROR_LIMIT = 100


def get_callback_error_limit() -> int:
    """Return the current callback error retention limit."""
    with _CALLBACK_ERROR_LIMIT_LOCK:
        return _CALLBACK_ERROR_LIMIT


Callback = Callable[["nx.Graph", dict[str, Any]], None]
CallbackRegistry = dict[str, dict[str, "CallbackSpec"]]


class CallbackError(TypedDict):
    """Metadata for a failed callback invocation."""

    event: str
    step: int | None
    error: str
    traceback: str
    fn: str
    name: str | None


def _func_id(fn: Callable[..., Any]) -> str:
    """Return a deterministic identifier for ``fn``.

    Combines the function's module and qualified name to avoid the
    nondeterminism of ``repr(fn)`` which includes the memory address.
    """
    module = getattr(fn, "__module__", fn.__class__.__module__)
    qualname = getattr(
        fn,
        "__qualname__",
        getattr(fn, "__name__", fn.__class__.__qualname__),
    )
    return f"{module}.{qualname}"


def _validate_registry(
    G: "nx.Graph", cbs: Any, dirty: set[str]
) -> CallbackRegistry:
    """Validate and normalise the callback registry.

    ``cbs`` is coerced to a ``defaultdict(dict)`` and any events listed in
    ``dirty`` are rebuilt using :func:`_normalize_callbacks`. Unknown events are
    removed. The cleaned registry is stored back on the graph and returned.
    """

    if not isinstance(cbs, Mapping):
        logger.warning(
            "Invalid callbacks registry on graph; resetting to empty",
        )
        cbs = defaultdict(dict)
    elif not isinstance(cbs, defaultdict) or cbs.default_factory is not dict:
        cbs = defaultdict(
            dict,
            {
                event: _normalize_callbacks(entries)
                for event, entries in dict(cbs).items()
                if event in _CALLBACK_EVENTS
            },
        )
    else:
        for event in dirty:
            if event in _CALLBACK_EVENTS:
                cbs[event] = _normalize_callbacks(cbs.get(event))
            else:
                cbs.pop(event, None)

    G.graph["callbacks"] = cbs
    return cbs


def _ensure_callbacks_nolock(G: "nx.Graph") -> CallbackRegistry:
    """Internal helper implementing ``_ensure_callbacks`` without locking."""
    cbs = G.graph.setdefault("callbacks", defaultdict(dict))
    dirty: set[str] = set(G.graph.pop("_callbacks_dirty", ()))
    return _validate_registry(G, cbs, dirty)


def _ensure_callbacks(G: "nx.Graph") -> CallbackRegistry:
    """Ensure the callback structure in ``G.graph``."""
    with _CALLBACK_LOCK:
        return _ensure_callbacks_nolock(G)


def _normalize_callbacks(entries: Any) -> dict[str, CallbackSpec]:
    """Return ``entries`` normalised into a callback mapping."""
    if isinstance(entries, Mapping):
        entries_iter = entries.values()
    elif isinstance(entries, Iterable) and not isinstance(entries, (str, bytes, bytearray)):
        entries_iter = entries
    else:
        return {}

    new_map: dict[str, CallbackSpec] = {}
    for entry in entries_iter:
        spec = _normalize_callback_entry(entry)
        if spec is None:
            continue
        key = spec.name or _func_id(spec.func)
        new_map[key] = spec
    return new_map


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
    elif is_non_string_sequence(entry):
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


def _record_callback_error(
    G: "nx.Graph",
    event: str,
    ctx: dict[str, Any],
    spec: CallbackSpec,
    err: Exception,
) -> None:
    """Log and store a callback error for later inspection.

    Errors are stored as :class:`CallbackError` entries inside
    ``G.graph['_callback_errors']``. The size of this deque is bounded by
    :func:`set_callback_error_limit`.
    """

    logger.exception("callback %r failed for %s: %s", spec.name, event, err)
    err_list = G.graph.get("_callback_errors")
    limit = get_callback_error_limit()
    if (
        not isinstance(err_list, deque)
        or err_list.maxlen != limit
    ):
        err_list = deque[CallbackError](maxlen=limit)
        G.graph["_callback_errors"] = err_list
    error: CallbackError = {
        "event": event,
        "step": ctx.get("step"),
        "error": repr(err),
        "traceback": traceback.format_exc(),
        "fn": _func_id(spec.func),
        "name": spec.name,
    }
    err_list.append(error)


def set_callback_error_limit(limit: int) -> int:
    """Set the maximum number of callback errors retained.

    Parameters
    ----------
    limit:
        Maximum number of recent callback errors to keep. Must be ``>= 1``.

    Returns
    -------
    int
        The previous limit.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    global _CALLBACK_ERROR_LIMIT
    with _CALLBACK_ERROR_LIMIT_LOCK:
        previous = _CALLBACK_ERROR_LIMIT
        _CALLBACK_ERROR_LIMIT = int(limit)
    return previous


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
    with _CALLBACK_LOCK:
        cbs = _ensure_callbacks_nolock(G)

        cb_name = name or getattr(func, "__name__", None)
        new_cb = CallbackSpec(cb_name, func)
        existing_map = cbs[event]
        cb_key = cb_name or _func_id(func)

        if cb_name is not None:
            existing_spec = existing_map.get(cb_key)
            if existing_spec is not None and existing_spec.func is not func:
                strict = bool(
                    G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"])
                )
                msg = f"Callback {cb_name!r} already registered for {event}"
                if strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
            # Explicit names override function identity when both are present.
            existing_map.pop(cb_key, None)
            fn_key = next(
                (k for k, spec in existing_map.items() if spec.func is func),
                None,
            )
            if fn_key is not None:
                existing_map.pop(fn_key, None)
        else:
            # Remove any existing registration by function identity when no
            # name is given.
            fn_key = next(
                (k for k, spec in existing_map.items() if spec.func is func),
                _func_id(func),
            )
            existing_map.pop(fn_key, None)

        existing_map[cb_key] = new_cb
        dirty = G.graph.setdefault("_callbacks_dirty", set())
        dirty.add(event)
    return func


def invoke_callbacks(
    G: "nx.Graph",
    event: CallbackEvent | str,
    ctx: dict[str, Any] | None = None,
) -> None:
    """Invoke all callbacks registered for ``event`` with context ``ctx``."""
    event = _normalize_event(event)
    with _CALLBACK_LOCK:
        cbs = dict(_ensure_callbacks_nolock(G).get(event, {}))
        strict = bool(
            G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"])
        )
    if ctx is None:
        ctx = {}
    for spec in cbs.values():
        try:
            spec.func(G, ctx)
        except (
            RuntimeError,
            ValueError,
            TypeError,
        ) as e:  # catch expected callback errors
            with _CALLBACK_LOCK:
                _record_callback_error(G, event, ctx, spec, e)
            if strict:
                raise
        except nx.NetworkXError as err:
            with _CALLBACK_LOCK:
                _record_callback_error(G, event, ctx, spec, err)
            # NetworkX errors are unexpected; log and re-raise
            logger.exception(
                "callback %r raised NetworkXError for %s with ctx=%r",
                spec.name,
                event,
                ctx,
            )
            raise
