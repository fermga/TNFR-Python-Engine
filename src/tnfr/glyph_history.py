"""Helpers for glyph history management."""

from __future__ import annotations

from typing import Dict, Any, Iterable
from collections import deque, Counter
import heapq
from itertools import islice

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
    window = int(window)
    if window < 0:
        raise ValueError("'window' must be non-negative")
    return window


def _ensure_glyph_history(nd: Dict[str, Any], window: int) -> deque:
    window = _validate_window(window)
    hist = nd.get("glyph_history")
    if not isinstance(hist, deque) or hist.maxlen != window:
        hist = deque(hist or [], maxlen=window)
        nd["glyph_history"] = hist
    return hist


def push_glyph(nd: Dict[str, Any], glyph: str, window: int) -> None:
    """Add ``glyph`` to node history with maximum size ``window``."""
    hist = _ensure_glyph_history(nd, window)
    hist.append(str(glyph))


def recent_glyph(nd: Dict[str, Any], glyph: str, window: int) -> bool:
    """Return ``True`` if ``glyph`` appeared in last ``window`` emissions."""
    gl = str(glyph)
    if window <= 0:
        if window < 0:
            _ensure_glyph_history(nd, window)
        return False
    hist = _ensure_glyph_history(nd, window)
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
        data: Dict[str, Any] | None = None,
        *,
        maxlen: int = 0,
        compact_every: int = 100,
    ) -> None:
        super().__init__(data or {})
        self._maxlen = maxlen
        self._compact_every = max(1, int(compact_every))
        self._counts: Dict[str, int] = {}
        self._heap: list[tuple[int, str]] = []
        if self._maxlen > 0:
            for k, v in list(self.items()):
                if isinstance(v, list):
                    super().__setitem__(k, deque(v, maxlen=self._maxlen))
                self._counts[k] = 0
                heapq.heappush(self._heap, (0, k))

    def _prune_heap(self) -> None:
        while self._heap and (
            self._heap[0][1] not in self
            or self._counts.get(self._heap[0][1]) != self._heap[0][0]
        ):
            heapq.heappop(self._heap)

    def _increment(self, key: str) -> None:
        cnt = self._counts.get(key, 0) + 1
        self._counts[key] = cnt
        heapq.heappush(self._heap, (cnt, key))
        if len(self._heap) > len(self) + self._compact_every:
            self._prune_heap()

    def _resolve_value(self, key: str, default: Any, *, insert: bool) -> Any:
        if insert:
            val = super().setdefault(key, default)
        else:
            val = super().__getitem__(key)
        if self._maxlen > 0 and isinstance(val, list):
            val = deque(val, maxlen=self._maxlen)
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
        self._counts.setdefault(key, 0)
        heapq.heappush(self._heap, (self._counts[key], key))
        if len(self._heap) > len(self) + self._compact_every:
            self._prune_heap()

    def setdefault(self, key, default=None):  # type: ignore[override]
        insert = key not in self
        val = self._resolve_value(key, default, insert=insert)
        if insert:
            self._counts[key] = 0
            heapq.heappush(self._heap, (0, key))
        return val

    def pop_least_used(self) -> Any:
        while self._heap:
            cnt, key = heapq.heappop(self._heap)
            if self._counts.get(key) == cnt and key in self:
                self._counts.pop(key, None)
                value = super().pop(key)
                self._prune_heap()
                return value
        raise KeyError("HistoryDict is empty; cannot pop least used")

    def pop_least_used_batch(self, k: int) -> None:
        if k <= 0:
            return
        self._prune_heap()
        removed = 0
        while self._heap and removed < k:
            cnt, key = heapq.heappop(self._heap)
            if self._counts.get(key) == cnt and key in self:
                self._counts.pop(key, None)
                super().pop(key, None)
                removed += 1
        self._prune_heap()


def ensure_history(G) -> Dict[str, Any]:
    """Ensure ``G.graph['history']`` exists and return it.

    ``HISTORY_MAXLEN`` must be non-negative and ``HISTORY_COMPACT_EVERY``
    must be a positive integer; otherwise a :class:`ValueError` is raised.
    When ``HISTORY_MAXLEN`` is zero, a plain dictionary is returned to avoid
    the overhead of :class:`HistoryDict`.
    """
    maxlen = int(get_param(G, "HISTORY_MAXLEN"))
    if maxlen < 0:
        raise ValueError("HISTORY_MAXLEN must be >= 0")
    compact_every = int(get_param(G, "HISTORY_COMPACT_EVERY"))
    if compact_every <= 0:
        raise ValueError("HISTORY_COMPACT_EVERY must be > 0")
    hist = G.graph.get("history")
    if maxlen == 0:
        if not isinstance(hist, dict) or isinstance(hist, HistoryDict):
            hist = dict(hist or {})
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


def append_metric(hist: Dict[str, Any], key: str, value: Any) -> None:
    """Append ``value`` to ``hist[key]`` list, creating it if missing."""
    if hasattr(hist, "get_increment"):
        lst = hist.get_increment(key, [])
    else:
        lst = hist.setdefault(key, [])
    lst.append(value)


def last_glyph(nd: Dict[str, Any]) -> str | None:
    """Return the most recent glyph for node or ``None``."""
    hist = nd.get("glyph_history")
    return hist[-1] if hist else None


def count_glyphs(
    G, window: int | None = None, *, last_only: bool = False
) -> Counter:
    """Count recent glyphs in the network.

    If ``window`` is ``None``, the full history for each node is used. When
    ``window`` is less than or equal to zero, no glyphs are counted for any
    node."""

    window_int = int(window) if window is not None else None
    if window_int is not None and window_int <= 0:
        return Counter()

    def _iter_seq(nd: Dict[str, Any]) -> Iterable[str]:
        if last_only:
            g = last_glyph(nd)
            if g:
                yield g
            return
        hist = nd.get("glyph_history")
        if not hist:
            return
        if window_int is None:
            yield from hist
        else:
            yield from islice(reversed(hist), window_int)

    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        counts.update(_iter_seq(nd))
    return counts
