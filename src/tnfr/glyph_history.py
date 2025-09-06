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


def push_glyph(nd: Dict[str, Any], glyph: str, window: int) -> None:
    """Add ``glyph`` to node history with maximum size ``window``."""
    if window < 0:
        raise ValueError("window must be >= 0")
    hist = nd.get("glyph_history")
    if hist is None or hist.maxlen != window:
        hist = deque(hist or [], maxlen=window)
        nd["glyph_history"] = hist
    hist.append(str(glyph))


def recent_glyph(nd: Dict[str, Any], glyph: str, window: int) -> bool:
    """Return ``True`` if ``glyph`` appeared in last ``window`` emissions."""
    gl = str(glyph)
    if window < 0:
        raise ValueError("window must be >= 0")
    if window == 0:
        return False

    hist = nd.get("glyph_history")
    if not hist:
        return False
    return gl in islice(reversed(hist), window)


class HistoryDict(dict):
    """Dict specialized for bounded history series and usage counts.

    Parameters
    ----------
    data:
        Initial mapping to populate the dictionary.
    maxlen:
        Maximum length for history lists stored as values.
    compact_every:
        Number of heap operations after which the internal heap is compacted.
        Higher values reduce the frequency of compaction. The default is 100.
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
        self._ops = 0
        self._dirty = 0
        self._counts: Dict[str, int] = {}
        self._heap: list[tuple[int, str]] = []
        if self._maxlen > 0:
            for k, v in list(self.items()):
                if isinstance(v, list):
                    self[k] = deque(v, maxlen=self._maxlen)
                self._counts.setdefault(k, 0)
                heapq.heappush(self._heap, (0, k))

    def _compact_heap(self) -> None:
        clean: list[tuple[int, str]] = []
        for cnt, k in self._heap:
            if k in self and self._counts.get(k) == cnt:
                clean.append((cnt, k))
        heapq.heapify(clean)
        self._heap = clean
        self._dirty = 0

    def _maybe_compact(self) -> None:
        self._ops += 1
        if self._ops >= self._compact_every and self._dirty > len(self):
            self._compact_heap()
            self._ops = 0

    def _increment(self, key: str) -> None:
        cnt = self._counts.get(key, 0) + 1
        self._counts[key] = cnt
        heapq.heappush(self._heap, (cnt, key))
        self._dirty += 1
        self._maybe_compact()

    def _get_and_increment(
        self, key: str, default: Any = None, *, missing: bool = False
    ):
        if not missing:
            val = super().__getitem__(key)
        else:
            val = super().setdefault(key, default)
            if self._maxlen > 0 and isinstance(val, list):
                val = deque(val, maxlen=self._maxlen)
                super().__setitem__(key, val)
        self._increment(key)
        return val

    def __getitem__(self, key):  # type: ignore[override]
        return self._get_and_increment(key)

    def get(self, key, default=None):  # type: ignore[override]
        try:
            return self._get_and_increment(key)
        except KeyError:
            return default

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        self._counts.setdefault(key, 0)
        heapq.heappush(self._heap, (self._counts[key], key))
        self._dirty += 1
        self._maybe_compact()

    def setdefault(self, key, default=None):  # type: ignore[override]
        if self._maxlen > 0 and isinstance(default, list):
            default = deque(default, maxlen=self._maxlen)
        if key in self:
            return self._get_and_increment(key)
        return self._get_and_increment(key, default, missing=True)

    def pop_least_used(self) -> Any:
        while True:
            try:
                cnt, key = heapq.heappop(self._heap)
            except IndexError as exc:
                raise KeyError("HistoryDict is empty; cannot pop least used") from exc
            if self._counts.get(key) == cnt and key in self:
                self._counts.pop(key, None)
                value = super().pop(key)
                self._maybe_compact()
                return value


def ensure_history(G) -> Dict[str, Any]:
    """Ensure ``G.graph['history']`` exists and return it.

    ``HISTORY_MAXLEN`` must be non-negative and ``HISTORY_COMPACT_EVERY``
    must be a positive integer; otherwise a :class:`ValueError` is raised.
    """
    maxlen = int(get_param(G, "HISTORY_MAXLEN"))
    if maxlen < 0:
        raise ValueError("HISTORY_MAXLEN must be >= 0")
    compact_every = int(get_param(G, "HISTORY_COMPACT_EVERY"))
    if compact_every <= 0:
        raise ValueError("HISTORY_COMPACT_EVERY must be > 0")
    hist = G.graph.get("history")
    if (
        not isinstance(hist, HistoryDict)
        or hist._maxlen != maxlen
        or hist._compact_every != compact_every
    ):
        hist = HistoryDict(hist, maxlen=maxlen, compact_every=compact_every)
        G.graph["history"] = hist
    if maxlen > 0:
        excess = len(hist) - maxlen
        if excess > 0:
            for _ in range(excess):
                hist.pop_least_used()
        # Note: trimming is O(n) only when history exceeds ``maxlen``
    return hist


def append_metric(hist: Dict[str, Any], key: str, value: Any) -> None:
    """Append ``value`` to ``hist[key]`` list, creating it if missing."""
    hist.setdefault(key, []).append(value)


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

    def _iter_seq(nd: Dict[str, Any]) -> Iterable[str]:
        if last_only:
            g = last_glyph(nd)
            return [g] if g else []
        hist = nd.get("glyph_history")
        if not hist:
            return []
        if window is None:
            return hist
        window_int = int(window)
        if window_int <= 0:
            return []
        return islice(reversed(hist), window_int)

    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        counts.update(_iter_seq(nd))
    return counts
