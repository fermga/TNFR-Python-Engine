"""Utilities for tracking structural operator emission history and related metrics.

This module tracks the history of glyphs (structural symbols like AL, EN, IL, etc.)
that are emitted when structural operators (Emission, Reception, Coherence, etc.)
are applied to nodes in the TNFR network.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, cast

from ._runtime_steps import (
    RUNTIME_STEP_NEXT_KEY,
    RUNTIME_STEP_UNSET,
    resolve_runtime_step_index,
)
from .constants import get_param, normalise_state_token
from .glyph_runtime import last_glyph
from .types import TNFRGraph

__all__ = (
    "HistoryDict",
    "push_glyph",
    "record_glyph_occurrence",
    "recent_glyph",
    "ensure_history",
    "current_step_idx",
    "current_operator_step",
    "next_operator_step",
    "append_metric",
    "count_glyphs",
)


def current_operator_step(nd: Mapping[str, Any]) -> int:
    """Return a monotonic operator index independent of bounded trace length."""

    from .operators.grammar_debt import read_grammar_state

    return read_grammar_state(nd).operator_step


def next_operator_step(nd: Mapping[str, Any]) -> int:
    """Return the index that the next canonical glyph will receive."""

    return current_operator_step(nd) + 1


def push_glyph(nd: MutableMapping[str, Any], glyph: str, window: int) -> None:
    """Add ``glyph`` to node history with maximum size ``window``.

    The shared window validator rejects invalid retention before recording.
    """

    from tnfr.validation.window import validate_window

    _record_glyph(nd, glyph, validate_window(window))


def record_glyph_occurrence(nd: MutableMapping[str, Any], glyph: Any) -> None:
    """Record one supplied occurrence, preserving the existing trace retention.

    A deque retains its declared maximum length; lists and tuples are retained
    as unbounded lists. Missing history starts an unbounded list. This only
    records supplied occurrence metadata: it neither executes nor authenticates
    an operator. Callers must not record an action already recorded by runtime.
    """

    _record_glyph(nd, glyph, None)


def _record_glyph(nd: MutableMapping[str, Any], glyph: Any, window: int | None) -> None:
    """Admit and record once; ``None`` preserves the current retention policy."""

    from .operators.grammar_debt import (
        OPERATOR_STEP_KEY,
        PRIOR_COHERENCE_KEY,
        U2_DEBT_KEY,
        advance_debt,
        advance_prior_coherence,
        canonical_operator_code,
        read_grammar_state,
    )

    # Capture debt before resizing/evicting the bounded trace. Neutral glyphs
    # and a zero-length trace cannot erase uncompensated destabilization.
    state = read_grammar_state(nd)
    # Freeze the monotonic index before a smaller window can evict retained
    # entries. Trace length is a fallback only; resizing must never move the
    # lifetime operator clock backwards.
    step = state.operator_step + 1
    next_debt = advance_debt(state.u2_debt, glyph)
    next_coherence = advance_prior_coherence(state.prior_coherence, glyph)
    recorded_glyph = canonical_operator_code(glyph)
    previous = nd.get("glyph_history", ())
    if window is None and isinstance(previous, deque):
        window = previous.maxlen
    if window is not None:
        if (
            isinstance(previous, deque)
            and previous.maxlen == window
            and all(
                type(item) is str and item == code
                for item, code in zip(previous, state.history)
            )
        ):
            hist = previous
        else:
            hist = deque(state.history, maxlen=window)
            nd["glyph_history"] = hist
    else:
        if isinstance(previous, list):
            hist = previous
            hist[:] = state.history
        elif isinstance(previous, deque):
            hist = previous
            hist.clear()
            hist.extend(state.history)
        else:
            hist = list(state.history)
        nd["glyph_history"] = hist
    hist.append(recorded_glyph)
    nd[OPERATOR_STEP_KEY] = step
    nd[U2_DEBT_KEY] = next_debt
    nd[PRIOR_COHERENCE_KEY] = next_coherence


def recent_glyph(nd: MutableMapping[str, Any], glyph: str, window: int) -> bool:
    """Return ``True`` if ``glyph`` appeared in last ``window`` emissions.

    This is a **read-only** operation that checks the existing history without
    modifying it. If ``window`` is zero, returns ``False``. Negative values
    raise :class:`ValueError`.

    Notes
    -----
    Reading a smaller window never truncates the retained deque.

    Reuses shared window, grammar-state and operator-identifier admission.
    """
    from tnfr.validation.window import validate_window

    from .operators.grammar_debt import canonical_operator_code, read_grammar_state

    v_window = validate_window(window)
    state = read_grammar_state(nd)
    code = canonical_operator_code(glyph)
    if v_window == 0:
        return False
    return code in state.history[-v_window:]


class HistoryDict(dict[str, Any]):
    """dict specialized for bounded history series and usage counts.

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
                if isinstance(v, (list, deque)):
                    super().__setitem__(k, self._to_deque(v))
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
        retained only when their bound already matches this history.
        """

        if isinstance(val, deque) and val.maxlen == self._maxlen:
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
    ``dict`` is used. The bound limits samples per series, never the number
    of metric streams. Older automatic least-used key eviction could remove
    unrelated diagnostics; callers needing key eviction must request it
    explicitly through :meth:`HistoryDict.pop_least_used`.
    """
    from .validation.window import validate_window

    maxlen = validate_window(get_param(G, "HISTORY_MAXLEN"))
    hist = G.graph.get("history")
    sentinel_key = "_metrics_history_id"
    replaced = False
    if maxlen == 0:
        if isinstance(hist, HistoryDict):
            hist = {
                key: list(value) if isinstance(value, deque) else value
                for key, value in hist.items()
            }
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
    if replaced:
        G.graph.pop(sentinel_key, None)
    _normalise_state_streams(cast(MutableMapping[str, Any], hist))
    return hist


def current_step_idx(G: TNFRGraph | Mapping[str, Any]) -> int:
    """Read the active runtime ordinal, or next ordinal outside a runtime call.

    Legacy standalone graphs without runtime markers retain their C_steps
    length fallback. That fallback is a retained sample index, not a lifetime
    execution count or physical time.
    """

    graph = getattr(G, "graph", G)
    next_index = graph.get(RUNTIME_STEP_NEXT_KEY, RUNTIME_STEP_UNSET)
    fallback = (
        len(graph.get("history", {}).get("C_steps", []))
        if next_index is RUNTIME_STEP_UNSET
        else 0
    )
    return resolve_runtime_step_index(next_index, fallback=fallback)


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
    """Count admitted recent glyphs by canonical code without rewriting traces.

    If ``window`` is ``None``, the full history for each node is used. A
    ``window`` of zero yields an empty :class:`Counter`. Negative values raise
    :class:`ValueError`.
    """

    from .operators.grammar_debt import read_grammar_state

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
        hist = read_grammar_state(nd).history
        if not hist:
            continue
        if window is None:
            seq = hist
        else:
            seq = hist[-window:]
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
