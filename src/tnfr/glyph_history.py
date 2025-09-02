"""Helpers for glyph history management."""
from __future__ import annotations

from typing import Dict, Any, Iterable
from collections import deque, Counter
import heapq
from itertools import islice

from .constants import ALIAS_EPI_KIND, get_param
from .helpers import get_attr_str

__all__ = [
    "HistoryDict",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "last_glyph",
    "count_glyphs",
]


def push_glyph(nd: Dict[str, Any], glyph: str, window: int) -> None:
    """Add ``glyph`` to node history with maximum size ``window``."""
    hist = nd.get("glyph_history")
    if hist is None or hist.maxlen != window:
        hist = deque(hist or [], maxlen=window)
        nd["glyph_history"] = hist
    hist.append(str(glyph))


def recent_glyph(nd: Dict[str, Any], glyph: str, ventana: int) -> bool:
    """Return ``True`` if ``glyph`` appeared in last ``ventana`` emissions."""
    gl = str(glyph)
    if ventana < 0:
        raise ValueError("ventana debe ser >= 0")

    last = last_glyph(nd)
    if ventana <= 1:
        return last == gl
    if last == gl:
        return True

    hist = nd.get("glyph_history")
    if hist:
        if hist and hist[-1] == last:
            history_iter = islice(reversed(hist), 1, ventana)
        else:
            history_iter = islice(reversed(hist), ventana - 1)
        for reciente in history_iter:
            if gl == reciente:
                return True
    return False


class HistoryDict(dict):
    """Dict specialized for bounded history series and usage counts."""

    def __init__(self, data: Dict[str, Any] | None = None, *, maxlen: int = 0):
        super().__init__(data or {})
        self._maxlen = maxlen
        self._counts: Dict[str, int] = {}
        self._heap: list[tuple[int, str]] = []
        if self._maxlen > 0:
            for k, v in list(self.items()):
                if isinstance(v, list):
                    self[k] = deque(v, maxlen=self._maxlen)
                self._counts.setdefault(k, 0)
                heapq.heappush(self._heap, (0, k))

    def _compact_heap(self) -> None:
        self._heap = [
            (cnt, k)
            for cnt, k in self._heap
            if k in self and self._counts.get(k) == cnt
        ]
        heapq.heapify(self._heap)

    def _maybe_compact(self) -> None:
        if len(self._heap) > len(self) * 2:
            self._compact_heap()

    def _increment(self, key: str) -> None:
        cnt = self._counts.get(key, 0) + 1
        self._counts[key] = cnt
        heapq.heappush(self._heap, (cnt, key))
        self._maybe_compact()

    def __getitem__(self, key):  # type: ignore[override]
        val = super().__getitem__(key)
        self._increment(key)
        return val

    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key, default)

    def tracked_get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        self._counts.setdefault(key, 0)
        heapq.heappush(self._heap, (self._counts[key], key))
        self._maybe_compact()

    def setdefault(self, key, default=None):  # type: ignore[override]
        if self._maxlen > 0 and isinstance(default, list):
            default = deque(default, maxlen=self._maxlen)
        if key in self:
            val = self[key]
        else:
            val = super().setdefault(key, default)
            if self._maxlen > 0 and isinstance(val, list):
                val = deque(val, maxlen=self._maxlen)
                super().__setitem__(key, val)
        self._increment(key)
        return val

    def pop_least_used(self) -> Any:
        while self._heap:
            cnt, key = heapq.heappop(self._heap)
            if self._counts.get(key) == cnt and key in self:
                self._counts.pop(key, None)
                return super().pop(key)
        raise KeyError("HistoryDict is empty")


def ensure_history(G) -> Dict[str, Any]:
    """Ensure ``G.graph['history']`` exists and return it."""
    maxlen = int(G.graph.get("HISTORY_MAXLEN", get_param(G, "HISTORY_MAXLEN")))
    hist = G.graph.get("history")
    if not isinstance(hist, HistoryDict) or hist._maxlen != maxlen:
        hist = HistoryDict(hist, maxlen=maxlen)
        G.graph["history"] = hist
    if maxlen > 0:
        while len(hist) > maxlen:
            try:
                hist.pop_least_used()
            except KeyError:
                break
    return hist


def last_glyph(nd: Dict[str, Any]) -> str | None:
    """Return the most recent glyph for node or ``None``."""
    kind = get_attr_str(nd, ALIAS_EPI_KIND, "")
    if kind:
        return kind
    hist = nd.get("glyph_history")
    if not hist:
        return None
    try:
        return hist[-1]
    except IndexError:
        return None


def count_glyphs(
    G, window: int | None = None, *, last_only: bool = False
) -> Counter:
    """Count recent glyphs in the network."""
    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        if last_only:
            g = last_glyph(nd)
            seq: Iterable[str] = [g] if g else []
        else:
            hist = nd.get("glyph_history")
            if not hist:
                continue
            if window is not None and window > 0:
                window_int = int(window)
                seq = islice(reversed(hist), window_int)
            else:
                seq = hist
        counts.update(seq)
    return counts

