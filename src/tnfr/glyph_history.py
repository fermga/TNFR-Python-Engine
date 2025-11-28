"""Utilities for tracking structural operator emission history and related metrics.

This module tracks the history of glyphs (structural symbols like AL, EN, IL, etc.)
that are emitted when structural operators (Emission, Reception, Coherence, etc.)
are applied to nodes in the TNFR network.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, MutableMapping
from itertools import islice
from typing import Any, cast

from .constants import get_param, normalise_state_token
from .glyph_runtime import last_glyph
from .types import TNFRGraph
from .utils import ensure_collection, get_logger

logger = get_logger(__name__)

__all__ = (
    "HistoryDict",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "current_step_idx",
    "append_metric",
    "count_glyphs",
)

_NU_F_HISTORY_KEYS = (
    "nu_f_rate_hz_str",
    "nu_f_rate_hz",
    "nu_f_ci_lower_hz_str",
    "nu_f_ci_upper_hz_str",
    "nu_f_ci_lower_hz",
    "nu_f_ci_upper_hz",
)


def _ensure_history(
    nd: MutableMapping[str, Any], window: int, *, create_zero: bool = False
) -> tuple[int, deque[str] | None]:
    """Validate ``window`` and ensure ``nd['glyph_history']`` deque."""

    from tnfr.validation.window import validate_window

    v_window = validate_window(window)
    if v_window == 0 and not create_zero:
        return v_window, None
    hist = nd.setdefault("glyph_history", deque(maxlen=v_window))
    if not isinstance(hist, deque) or hist.maxlen != v_window:
        # Rebuild deque from any iterable, ignoring raw strings/bytes and scalars
        if isinstance(hist, (str, bytes, bytearray)):
            items: Iterable[Any] = ()
        else:
            try:
                items = ensure_collection(hist, max_materialize=None)
            except TypeError:
                logger.debug("Discarding non-iterable glyph history value %r", hist)
                items = ()
        hist = deque((str(item) for item in items), maxlen=v_window)
        nd["glyph_history"] = hist
    return v_window, hist


def push_glyph(nd: MutableMapping[str, Any], glyph: str, window: int) -> None:
    """Add ``glyph`` to node history with maximum size ``window``.

    ``window`` validation and deque creation are handled by
    :func:`_ensure_history`.
    """

    _, hist = _ensure_history(nd, window, create_zero=True)
    hist.append(str(glyph))


def recent_glyph(nd: MutableMapping[str, Any], glyph: str, window: int) -> bool:
    """Return ``True`` if ``glyph`` appeared in last ``window`` emissions.

    This is a **read-only** operation that checks the existing history without
    modifying it. If ``window`` is zero, returns ``False``. Negative values
    raise :class:`ValueError`.

    Notes
    -----
    This function intentionally does NOT call ``_ensure_history`` to avoid
    accidentally truncating the glyph_history deque when checking with a
    smaller window than the deque's maxlen. This preserves the canonical
    principle that reading history should not modify it.

    Reuses ``validate_window`` and ``ensure_collection`` utilities.
    """
    from tnfr.validation.window import validate_window

    v_window = validate_window(window)
    if v_window == 0:
        return False

    # Read existing history without modifying it
    hist = nd.get("glyph_history")
    if hist is None:
        return False

    gl = str(glyph)

    # Use canonical ensure_collection to materialize history
    try:
        items = list(ensure_collection(hist, max_materialize=None))
    except (TypeError, ValueError):
        return False

    # Check only the last v_window items
    recent_items = items[-v_window:] if len(items) > v_window else items
    return gl in recent_items


class HistoryDict(dict[str, Any]):
    """Dict specialized for bounded history series and usage counts.

    Usage counts are tracked explicitly via :meth:`get_increment`. Accessing
    keys through ``__getitem__`` or :meth:`get` does not affect the internal
    counters, avoiding surprising evictions on mere reads. Counting is now
    handled with :class:`collections.Counter` alone, relying on
    :meth:`Counter.most_common` to locate least-used entries when required.

    Parameters
    ----------
    data:
        Initial mapping to populate the dictionary.
    maxlen:
        Maximum length for history lists stored as values.
    """

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        *,
        maxlen: int = 0,
    ) -> None:
        super().__init__(data or {})
        self._maxlen = maxlen
        self._counts: Counter[str] = Counter()
        if self._maxlen > 0:
            for k, v in list(self.items()):
                if isinstance(v, list):
                    super().__setitem__(k, deque(v, maxlen=self._maxlen))
                self._counts[k] = 0
        else:
            for k in self:
                self._counts[k] = 0
        # ``_heap`` is no longer required with ``Counter.most_common``.

    def _increment(self, key: str) -> None:
        """Increase usage count for ``key``."""
        self._counts[key] += 1

    def _to_deque(self, val: Any) -> deque[Any]:
        """Coerce ``val`` to a deque respecting ``self._maxlen``.

        ``Iterable`` inputs (excluding ``str`` and ``bytes``) are expanded into
        the deque, while single values are wrapped. Existing deques are
        returned unchanged.
        """

        if isinstance(val, deque):
            return val
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            return deque(val, maxlen=self._maxlen)
        return deque([val], maxlen=self._maxlen)

    def _resolve_value(self, key: str, default: Any, *, insert: bool) -> Any:
        if insert:
            val = super().setdefault(key, default)
        else:
            val = super().__getitem__(key)
        if self._maxlen > 0:
            if not isinstance(val, Mapping):
                val = self._to_deque(val)
            super().__setitem__(key, val)
        return val

    def get_increment(self, key: str, default: Any = None) -> Any:
        """Return value for ``key`` and increment its usage counter."""

        insert = key not in self
        val = self._resolve_value(key, default, insert=insert)
        self._increment(key)
        return val

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        """Return the tracked value for ``key`` ensuring deque normalisation."""

        return self._resolve_value(key, None, insert=False)

    def get(self, key: str, default: Any | None = None) -> Any:  # type: ignore[override]
        """Return ``key`` when present; otherwise fall back to ``default``."""

        try:
            return self._resolve_value(key, None, insert=False)
        except KeyError:
            return default

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        """Store ``value`` for ``key`` while initialising usage tracking."""

        super().__setitem__(key, value)
        if key not in self._counts:
            self._counts[key] = 0

    def setdefault(self, key: str, default: Any | None = None) -> Any:  # type: ignore[override]
        """Return existing value for ``key`` or insert ``default`` when absent."""

        insert = key not in self
        val = self._resolve_value(key, default, insert=insert)
        if insert:
            self._counts[key] = 0
        return val

    def pop_least_used(self) -> Any:
        """Remove and return the value with the smallest usage count."""
        while self._counts:
            key = min(self._counts, key=self._counts.get)
            self._counts.pop(key, None)
            if key in self:
                return super().pop(key)
        raise KeyError("HistoryDict is empty; cannot pop least used")

    def pop_least_used_batch(self, k: int) -> None:
        """Remove up to ``k`` least-used entries from the history."""

        for _ in range(max(0, int(k))):
            try:
                self.pop_least_used()
            except KeyError:
                break


def ensure_history(G: TNFRGraph) -> HistoryDict | dict[str, Any]:
    """Ensure ``G.graph['history']`` exists and return it.

    ``HISTORY_MAXLEN`` must be non-negative; otherwise a
    :class:`ValueError` is raised. When ``HISTORY_MAXLEN`` is zero, a regular
    ``dict`` is used.
    """
    maxlen, _ = _ensure_history({}, int(get_param(G, "HISTORY_MAXLEN")))
    hist = G.graph.get("history")
    sentinel_key = "_metrics_history_id"
    replaced = False
    if maxlen == 0:
        if isinstance(hist, HistoryDict):
            hist = dict(hist)
            G.graph["history"] = hist
            replaced = True
        elif hist is None:
            hist = {}
            G.graph["history"] = hist
            replaced = True
        if replaced:
            G.graph.pop(sentinel_key, None)
        if isinstance(hist, MutableMapping):
            _normalise_state_streams(hist)
        return hist
    if not isinstance(hist, HistoryDict) or hist._maxlen != maxlen:
        hist = HistoryDict(hist, maxlen=maxlen)
        G.graph["history"] = hist
        replaced = True
    excess = len(hist) - maxlen
    if excess > 0:
        hist.pop_least_used_batch(excess)
    if replaced:
        G.graph.pop(sentinel_key, None)
    _normalise_state_streams(cast(MutableMapping[str, Any], hist))
    return hist


def current_step_idx(G: TNFRGraph | Mapping[str, Any]) -> int:
    """Return the current step index from ``G`` history."""

    graph = getattr(G, "graph", G)
    return len(graph.get("history", {}).get("C_steps", []))


def append_metric(hist: MutableMapping[str, list[Any]], key: str, value: Any) -> None:
    """Append ``value`` to ``hist[key]`` list, creating it if missing."""
    if key == "phase_state" and isinstance(value, str):
        value = normalise_state_token(value)
    elif key == "nodal_diag" and isinstance(value, Mapping):
        snapshot: dict[Any, Any] = {}
        for node, payload in value.items():
            if isinstance(payload, Mapping):
                state_value = payload.get("state")
                if isinstance(payload, MutableMapping):
                    updated = payload
                else:
                    updated = dict(payload)
                if isinstance(state_value, str):
                    updated["state"] = normalise_state_token(state_value)
                snapshot[node] = updated
            else:
                snapshot[node] = payload
        hist.setdefault(key, []).append(snapshot)
        return

    hist.setdefault(key, []).append(value)


def count_glyphs(
    G: TNFRGraph, window: int | None = None, *, last_only: bool = False
) -> Counter[str]:
    """Count recent glyphs in the network.

    If ``window`` is ``None``, the full history for each node is used. A
    ``window`` of zero yields an empty :class:`Counter`. Negative values raise
    :class:`ValueError`.
    """

    if window is not None:
        from tnfr.validation.window import validate_window

        window = validate_window(window)
        if window == 0:
            return Counter()

    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        if last_only:
            g = last_glyph(nd)
            if g:
                counts[g] += 1
            continue
        hist = nd.get("glyph_history")
        if not hist:
            continue
        if window is None:
            seq = hist
        else:
            start = max(len(hist) - window, 0)
            seq = islice(hist, start, None)
        counts.update(seq)

    return counts


def _normalise_state_streams(hist: MutableMapping[str, Any]) -> None:
    """Normalise legacy state tokens stored in telemetry history."""

    phase_state = hist.get("phase_state")
    if isinstance(phase_state, deque):
        canonical = [normalise_state_token(str(item)) for item in phase_state]
        if canonical != list(phase_state):
            phase_state.clear()
            phase_state.extend(canonical)
    elif isinstance(phase_state, list):
        canonical = [normalise_state_token(str(item)) for item in phase_state]
        if canonical != phase_state:
            hist["phase_state"] = canonical

    diag_history = hist.get("nodal_diag")
    if isinstance(diag_history, list):
        for snapshot in diag_history:
            if not isinstance(snapshot, Mapping):
                continue
            for node, payload in snapshot.items():
                if not isinstance(payload, Mapping):
                    continue
                state_value = payload.get("state")
                if not isinstance(state_value, str):
                    continue
                canonical = normalise_state_token(state_value)
                if canonical == state_value:
                    continue
                if isinstance(payload, MutableMapping):
                    payload["state"] = canonical
                else:
                    snapshot[node] = {**payload, "state": canonical}
