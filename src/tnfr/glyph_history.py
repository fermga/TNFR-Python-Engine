"""Utilities for tracking glyph emission history and related metrics."""

from __future__ import annotations

from typing import Any, Protocol
from collections import deque, Counter
from itertools import islice
import heapq
from collections.abc import Iterable

from .constants import get_param

__all__ = [
    "HistoryDict",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "append_metric",
    "last_glyph",
    "count_glyphs",
]


def _validate_window(window: int) -> int:
    """Validate and coerce ``window`` to a non-negative ``int``."""

    window_int = int(window)
    if window_int < 0:
        raise ValueError("'window' must be non-negative")
    return window_int


def _ensure_glyph_history(nd: dict[str, Any], window: int) -> deque:
    """Return ``nd['glyph_history']`` deque after validating ``window``.

    Non-iterable existing values are discarded.
    """

    window_int = _validate_window(window)
    hist = nd.get("glyph_history")
    if not isinstance(hist, deque) or hist.maxlen != window_int:
        seq = (
            hist
            if isinstance(hist, Iterable) and not isinstance(hist, (str, bytes))
            else []
        )
        hist = deque(seq, maxlen=window_int)
        nd["glyph_history"] = hist
    return hist


def push_glyph(nd: dict[str, Any], glyph: str, window: int) -> None:
    """Add ``glyph`` to node history with maximum size ``window``.

    ``window`` is validated and the underlying deque is ensured by
    :func:`_ensure_glyph_history`.
    """

    hist = _ensure_glyph_history(nd, window)
    hist.append(str(glyph))


def recent_glyph(nd: dict[str, Any], glyph: str, window: int) -> bool:
    """Return ``True`` if ``glyph`` appeared in last ``window`` emissions.

    ``window`` is validated and the history deque ensured internally. A
    ``window`` of zero returns ``False`` without modifying ``nd``. Negative
    values raise :class:`ValueError`.
    """
    window_int = _validate_window(window)
    if window_int == 0:
        return False
    hist = _ensure_glyph_history(nd, window_int)
    gl = str(glyph)
    return gl in hist


class HistoryDict(dict):
    """Dict specialized for bounded history series and usage counts.

    Usage counts are tracked explicitly via :meth:`get_increment`. Accessing
    keys through ``__getitem__`` or :meth:`get` does not affect the internal
    counters, avoiding surprising evictions on mere reads. Stale entries are
    periodically discarded to keep the heap size under control without
    maintaining explicit dirtiness counters.

    Parameters
    ----------
    data:
        Initial mapping to populate the dictionary.
    maxlen:
        Maximum length for history lists stored as values.
    compact_every:
        Additional slack allowed in the internal heap before pruning stale
        entries. Larger values reduce the frequency of pruning. The default
        is 100.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        maxlen: int = 0,
        compact_every: int = 100,
    ) -> None:
        super().__init__(data or {})
        self._maxlen = maxlen
        self._compact_every = max(1, int(compact_every))
        self._counts: Counter[str] = Counter()
        self._heap: list[tuple[int, str]] = []
        self._heap_index: dict[str, int] = {}
        if self._maxlen > 0:
            for k, v in list(self.items()):
                if isinstance(v, list):
                    super().__setitem__(k, deque(v, maxlen=self._maxlen))
                self._counts[k] = 0
        else:
            for k in self:
                self._counts[k] = 0
        self._heap = [(cnt, k) for k, cnt in self._counts.items()]
        heapq.heapify(self._heap)
        self._heap_index = {k: i for i, (_, k) in enumerate(self._heap)}

    def _increment(self, key: str) -> None:
        self._counts[key] += 1
        heapq.heappush(self._heap, (self._counts[key], key))
        self._heap_index = {k: i for i, (_, k) in enumerate(self._heap)}
        self._prune_heap()

    def _prune_heap(self) -> None:
        """Ensure heap size stays within ``target`` keeping valid entries."""
        target = len(self._counts) + self._compact_every
        if len(self._heap) <= target:
            return
        self._heap = heapq.nsmallest(
            target,
            (
                (cnt, key)
                for cnt, key in self._heap
                if self._counts.get(key) == cnt
            ),
        )
        heapq.heapify(self._heap)
        self._heap_index = {k: i for i, (_, k) in enumerate(self._heap)}

    def _pop_heap_key(self) -> str:
        """Pop and return the key with the smallest count from the heap."""
        while self._heap:
            cnt, key = heapq.heappop(self._heap)
            if self._counts.get(key) == cnt:
                self._heap_index = {k: i for i, (_, k) in enumerate(self._heap)}
                return key
        self._heap_index = {k: i for i, (_, k) in enumerate(self._heap)}
        raise KeyError("HistoryDict is empty; cannot pop least used")

    def _to_deque(self, val: Any) -> deque:
        """Coerce ``val`` to a deque respecting ``self._maxlen``.

        ``Iterable`` inputs (excluding ``str`` and ``bytes``) are expanded into
        the deque, while single values are wrapped. Existing deques are returned
        unchanged.
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
            val = self._to_deque(val)
            super().__setitem__(key, val)
        return val

    def get_increment(self, key: str, default: Any = None) -> Any:
        insert = key not in self
        val = self._resolve_value(key, default, insert=insert)
        self._increment(key)
        return val

    def __getitem__(self, key):  # type: ignore[override]
        return self._resolve_value(key, None, insert=False)

    def get(self, key, default=None):  # type: ignore[override]
        try:
            return self._resolve_value(key, None, insert=False)
        except KeyError:
            return default

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        if key in self._counts:
            idx = self._heap_index.get(key)
            if idx is not None:
                self._heap[idx] = (self._counts[key], key)
                heapq._siftup(self._heap, idx)
                heapq._siftdown(self._heap, 0, idx)
            else:
                heapq.heappush(self._heap, (self._counts[key], key))
        else:
            self._counts[key] = 0
            heapq.heappush(self._heap, (0, key))
        self._heap_index = {k: i for i, (_, k) in enumerate(self._heap)}
        self._prune_heap()

    def setdefault(self, key, default=None):  # type: ignore[override]
        insert = key not in self
        val = self._resolve_value(key, default, insert=insert)
        if insert:
            self._counts[key] = 0
            heapq.heappush(self._heap, (0, key))
            self._heap_index = {k: i for i, (_, k) in enumerate(self._heap)}
            self._prune_heap()
        return val

    def pop_least_used(self) -> Any:
        while self._counts:
            key = self._pop_heap_key()
            self._counts.pop(key, None)
            if key in self:
                return super().pop(key)
        raise KeyError("HistoryDict is empty; cannot pop least used")

    def pop_least_used_batch(self, k: int) -> None:
        for _ in range(max(0, int(k))):
            try:
                self.pop_least_used()
            except KeyError:
                break


def ensure_history(G) -> dict[str, Any]:
    """Ensure ``G.graph['history']`` exists and return it.

    ``HISTORY_MAXLEN`` must be non-negative and ``HISTORY_COMPACT_EVERY``
    must be a positive integer; otherwise a :class:`ValueError` is raised.
    When ``HISTORY_MAXLEN`` is zero, a lightweight dictionary subclass is
    returned to avoid the overhead of :class:`HistoryDict` while still
    exposing :meth:`get_increment`.
    """
    maxlen = int(get_param(G, "HISTORY_MAXLEN"))
    if maxlen < 0:
        raise ValueError("HISTORY_MAXLEN must be >= 0")
    compact_every = int(get_param(G, "HISTORY_COMPACT_EVERY"))
    if compact_every <= 0:
        raise ValueError("HISTORY_COMPACT_EVERY must be > 0")
    hist = G.graph.get("history")
    if maxlen == 0:
        if not isinstance(hist, _IncrementDict):
            hist = _IncrementDict(hist or {})
            G.graph["history"] = hist
        return hist
    if (
        not isinstance(hist, HistoryDict)
        or hist._maxlen != maxlen
        or hist._compact_every != compact_every
    ):
        hist = HistoryDict(hist, maxlen=maxlen, compact_every=compact_every)
        G.graph["history"] = hist
    excess = len(hist) - maxlen
    if excess > 0:
        hist.pop_least_used_batch(excess)
    return hist


class _IncrementDict(dict):
    """Dict with ``get_increment`` for metric history."""

    def get_increment(
        self, key: str, default: Any = None
    ) -> Any:  # noqa: D401
        return self.setdefault(key, default)


class SupportsGetIncrement(Protocol):
    def get_increment(self, key: str, default: Any | None = None) -> Any:
        """Return value for *key*, inserting *default* if missing."""


class _IncrementProxy:
    """Adapter to provide :meth:`get_increment` for plain dictionaries."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get_increment(self, key: str, default: Any | None = None) -> Any:
        return self._data.setdefault(key, default)


def _ensure_increment(
    hist: dict[str, Any] | SupportsGetIncrement,
) -> SupportsGetIncrement:
    return (
        hist
        if callable(getattr(hist, "get_increment", None))
        else _IncrementProxy(hist)
    )  # type: ignore[return-value]


def append_metric(
    hist: dict[str, Any] | SupportsGetIncrement, key: str, value: Any
) -> None:
    """Append ``value`` to ``hist[key]`` list, creating it if missing."""
    _ensure_increment(hist).get_increment(key, []).append(value)


def last_glyph(nd: dict[str, Any]) -> str | None:
    """Return the most recent glyph for node or ``None``."""
    hist = nd.get("glyph_history")
    return hist[-1] if hist else None


def count_glyphs(
    G, window: int | None = None, *, last_only: bool = False
) -> Counter:
    """Count recent glyphs in the network.

    If ``window`` is ``None``, the full history for each node is used. A
    ``window`` of zero yields an empty :class:`Counter`. Negative values raise
    :class:`ValueError`.
    """

    if window is not None:
        window_int = _validate_window(window)
        if window_int == 0:
            return Counter()
    else:
        window_int = None

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
        if window_int is None:
            seq = hist
        else:
            start = max(len(hist) - window_int, 0)
            seq = islice(hist, start, None)
        counts.update(seq)

    return counts
