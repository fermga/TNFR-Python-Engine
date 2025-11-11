"""Callback registration and invocation helpers.

This module is thread-safe: all mutations of the callback registry stored in a
graph's ``G.graph`` are serialised using a process-wide lock obtained via
``locking.get_lock("callbacks")``. Callback functions themselves execute
outside of the lock and must therefore be independently thread-safe if they
modify shared state.
"""

from __future__ import annotations

import threading
import traceback
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import Any, NamedTuple

import networkx as nx

from ..constants import DEFAULTS
from ..locking import get_lock
from .init import get_logger
from .data import is_non_string_sequence
from ..types import CallbackError


class CallbackSpec(NamedTuple):
    """Specification for a registered callback."""

    name: str | None
    func: Callable[..., Any]


__all__ = (
    "CallbackEvent",
    "CallbackManager",
    "callback_manager",
    "CallbackError",
    "CallbackSpec",
)

logger = get_logger(__name__)


class CallbackEvent(str, Enum):
    """Supported callback events."""

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_REMESH = "on_remesh"
    CACHE_METRICS = "cache_metrics"


class CallbackManager:
    """Centralised registry and error tracking for callbacks."""

    def __init__(self) -> None:
        self._lock = get_lock("callbacks")
        self._error_limit_lock = threading.Lock()
        self._error_limit = 100
        self._error_limit_cache = self._error_limit

    # ------------------------------------------------------------------
    # Error limit management
    # ------------------------------------------------------------------
    def get_callback_error_limit(self) -> int:
        """Return the current callback error retention limit."""
        with self._error_limit_lock:
            return self._error_limit

    def set_callback_error_limit(self, limit: int) -> int:
        """Set the maximum number of callback errors retained."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._error_limit_lock:
            previous = self._error_limit
            self._error_limit = int(limit)
            self._error_limit_cache = self._error_limit
        return previous

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def _record_callback_error(
        self,
        G: "nx.Graph",
        event: str,
        ctx: dict[str, Any],
        spec: CallbackSpec,
        err: Exception,
    ) -> None:
        """Log and store a callback error for later inspection."""

        logger.exception("callback %r failed for %s: %s", spec.name, event, err)
        limit = self._error_limit_cache
        err_list = G.graph.setdefault("_callback_errors", deque[CallbackError](maxlen=limit))
        if err_list.maxlen != limit:
            err_list = deque[CallbackError](err_list, maxlen=limit)
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

    def _ensure_callbacks_nolock(self, G: "nx.Graph") -> CallbackRegistry:
        cbs = G.graph.setdefault("callbacks", defaultdict(dict))
        dirty: set[str] = set(G.graph.pop("_callbacks_dirty", ()))
        return _validate_registry(G, cbs, dirty)

    def _ensure_callbacks(self, G: "nx.Graph") -> CallbackRegistry:
        with self._lock:
            return self._ensure_callbacks_nolock(G)

    def register_callback(
        self,
        G: "nx.Graph",
        event: CallbackEvent | str,
        func: Callback,
        *,
        name: str | None = None,
    ) -> Callback:
        """Register ``func`` as callback for ``event``."""

        event = _normalize_event(event)
        _ensure_known_event(event)
        if not callable(func):
            raise TypeError("func must be callable")
        with self._lock:
            cbs = self._ensure_callbacks_nolock(G)

            cb_name = name or getattr(func, "__name__", None)
            spec = CallbackSpec(cb_name, func)
            existing_map = cbs[event]
            strict = bool(G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"]))
            key = _reconcile_callback(event, existing_map, spec, strict)

            existing_map[key] = spec
            dirty = G.graph.setdefault("_callbacks_dirty", set())
            dirty.add(event)
        return func

    def invoke_callbacks(
        self,
        G: "nx.Graph",
        event: CallbackEvent | str,
        ctx: dict[str, Any] | None = None,
    ) -> None:
        """Invoke all callbacks registered for ``event`` with context ``ctx``."""

        event = _normalize_event(event)
        with self._lock:
            cbs = dict(self._ensure_callbacks_nolock(G).get(event, {}))
            strict = bool(G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"]))
        if ctx is None:
            ctx = {}
        for spec in cbs.values():
            try:
                spec.func(G, ctx)
            except (
                RuntimeError,
                ValueError,
                TypeError,
            ) as e:
                with self._lock:
                    self._record_callback_error(G, event, ctx, spec, e)
                if strict:
                    raise
            except nx.NetworkXError as err:
                with self._lock:
                    self._record_callback_error(G, event, ctx, spec, err)
                logger.exception(
                    "callback %r raised NetworkXError for %s with ctx=%r",
                    spec.name,
                    event,
                    ctx,
                )
                raise


Callback = Callable[["nx.Graph", dict[str, Any]], None]
CallbackRegistry = dict[str, dict[str, "CallbackSpec"]]


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


def _validate_registry(G: "nx.Graph", cbs: Any, dirty: set[str]) -> CallbackRegistry:
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
                if _is_known_event(event)
            },
        )
    else:
        for event in dirty:
            if _is_known_event(event):
                cbs[event] = _normalize_callbacks(cbs.get(event))
            else:
                cbs.pop(event, None)

    G.graph["callbacks"] = cbs
    return cbs


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


def _is_known_event(event: str) -> bool:
    """Return ``True`` when ``event`` matches a declared :class:`CallbackEvent`."""

    try:
        CallbackEvent(event)
    except ValueError:
        return False
    else:
        return True


def _ensure_known_event(event: str) -> None:
    """Raise :class:`ValueError` when ``event`` is not a known callback."""

    try:
        CallbackEvent(event)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown event: {event}") from exc


def _normalize_callback_entry(entry: Any) -> "CallbackSpec | None":
    """Normalize a callback specification.

    Supported formats
    -----------------
    * :class:`CallbackSpec` instances (returned unchanged).
    * Sequences ``(name: str, func: Callable)`` such as lists, tuples or other
      iterables.
    * Bare callables ``func`` whose name is taken from ``func.__name__``.

    ``None`` is returned when ``entry`` does not match any of the accepted
    formats. The original ``entry`` is never mutated. Sequence inputs are
    converted to ``tuple`` before validation to support generators; the
    materialization consumes the iterable and failure results in ``None``.
    """

    if isinstance(entry, CallbackSpec):
        return entry
    elif is_non_string_sequence(entry):
        try:
            entry = tuple(entry)
        except TypeError:
            return None
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


def _reconcile_callback(
    event: str,
    existing_map: dict[str, CallbackSpec],
    spec: CallbackSpec,
    strict: bool,
) -> str:
    """Reconcile ``spec`` with ``existing_map``.

    Ensures that callbacks remain unique by explicit name or function identity.
    When a name collision occurs with a different function, ``strict`` controls
    whether a :class:`ValueError` is raised or a warning is logged.

    Parameters
    ----------
    event:
        Event under which ``spec`` will be registered. Only used for messages.
    existing_map:
        Current mapping of callbacks for ``event``.
    spec:
        Callback specification being registered.
    strict:
        Whether to raise on name collisions instead of logging a warning.

    Returns
    -------
    str
        Key under which ``spec`` should be stored in ``existing_map``.
    """

    key = spec.name or _func_id(spec.func)

    if spec.name is not None:
        existing_spec = existing_map.get(key)
        if existing_spec is not None and existing_spec.func is not spec.func:
            msg = f"Callback {spec.name!r} already registered for {event}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

    # Remove existing entries under the same key and any other using the same
    # function identity to avoid duplicates.
    existing_map.pop(key, None)
    fn_key = next((k for k, s in existing_map.items() if s.func is spec.func), None)
    if fn_key is not None:
        existing_map.pop(fn_key, None)

    return key


# ---------------------------------------------------------------------------
# Default manager instance and convenience wrappers
# ---------------------------------------------------------------------------

callback_manager = CallbackManager()
