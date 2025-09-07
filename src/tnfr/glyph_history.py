"""Helpers for glyph history management."""

from __future__ import annotations

from typing import Dict, Any, Iterable
from collections import deque, Counter
from itertools import islice
import heapq

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
    window = _validate_window(window)
    if window == 0:
        return False
    gl = str(glyph)
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
        self._counts: Counter[str] = Counter()
        if self._maxlen > 0:
            for k, v in list(self.items()):
                if isinstance(v, list):
                    super().__setitem__(k, deque(v, maxlen=self._maxlen))
                self._counts[k] = 0

    def _increment(self, key: str) -> None:
        self._counts[key] += 1

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

    def setdefault(self, key, default=None):  # type: ignore[override]
        insert = key not in self
        val = self._resolve_value(key, default, insert=insert)
        if insert:
            self._counts[key] = 0
        return val

    def pop_least_used(self) -> Any:
        while self._counts:
            key = min(self._counts, key=self._counts.get)
            self._counts.pop(key, None)
            if key in self:
                return super().pop(key)
        raise KeyError("HistoryDict is empty; cannot pop least used")

    def pop_least_used_batch(self, k: int) -> None:
        if k > 0 and self._counts:
            removed = 0
            for key, _ in heapq.nsmallest(
                len(self._counts), self._counts.items(), key=lambda item: item[1]
            ):
                self._counts.pop(key, None)
                if key in self:
                    super().pop(key, None)
                    removed += 1
                    if removed >= k:
                        break


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

    return Counter(
        g for _, nd in G.nodes(data=True) for g in _iter_seq(nd)
    )
