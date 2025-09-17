"""Flattening utilities to compile TNFR token sequences."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import Any, Callable

from .collections_utils import (
    MAX_MATERIALIZE_DEFAULT,
    ensure_collection,
    flatten_structure,
    normalize_materialize_limit,
)
from .constants_glyphs import GLYPHS_CANONICAL_SET
from .tokens import THOL, TARGET, WAIT, OpTag, THOL_SENTINEL, Token
from .types import Glyph

__all__ = [
    "THOLEvaluator",
    "_flatten",
    "_flatten_glyph",
    "_flatten_target",
    "_flatten_wait",
]


_STRING_TYPES = (str, bytes, bytearray)


def _iter_source(
    seq: Iterable[Token] | Sequence[Token] | Any,
    *,
    max_materialize: int | None,
) -> Iterable[Any]:
    """Yield items from ``seq`` enforcing ``max_materialize`` when needed."""

    if isinstance(seq, Collection) and not isinstance(seq, _STRING_TYPES):
        return seq

    if isinstance(seq, _STRING_TYPES):
        return (seq,)

    if not isinstance(seq, Iterable):
        raise TypeError(f"{seq!r} is not iterable")

    limit = normalize_materialize_limit(max_materialize)
    if limit is None:
        return seq
    if limit == 0:
        return ()

    def _limited() -> Iterator[Any]:
        samples: list[Any] = []
        for idx, item in enumerate(seq, 1):
            if len(samples) < 3:
                samples.append(item)
            if idx > limit:
                examples = ", ".join(repr(x) for x in samples)
                raise ValueError(
                    "Iterable produced "
                    f"{idx} items, exceeds limit {limit}; first items: [{examples}]"
                )
            yield item

    return _limited()


def _push_thol_frame(
    frames: list[dict[str, Any]],
    item: THOL,
    *,
    max_materialize: int | None,
) -> None:
    """Validate ``item`` and append a frame for its evaluation."""

    repeats = int(item.repeat)
    if repeats < 1:
        raise ValueError("repeat must be ≥1")
    if item.force_close is not None and not isinstance(item.force_close, Glyph):
        raise ValueError("force_close must be a Glyph")
    closing = (
        item.force_close
        if isinstance(item.force_close, Glyph)
        and item.force_close in {Glyph.SHA, Glyph.NUL}
        else None
    )
    seq0 = ensure_collection(
        item.body,
        max_materialize=max_materialize,
        error_msg=f"THOL body exceeds max_materialize={max_materialize}",
    )
    frames.append(
        {
            "seq": seq0,
            "index": 0,
            "remaining": repeats,
            "closing": closing,
        }
    )


class THOLEvaluator:
    """Generator that expands a :class:`THOL` block lazily."""

    def __init__(
        self,
        item: THOL,
        *,
        max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
    ) -> None:
        self._frames: list[dict[str, Any]] = []
        _push_thol_frame(self._frames, item, max_materialize=max_materialize)
        self._max_materialize = max_materialize
        self._started = False

    def __iter__(self) -> "THOLEvaluator":
        return self

    def __next__(self):
        if not self._started:
            self._started = True
            return THOL_SENTINEL
        while self._frames:
            frame = self._frames[-1]
            seq = frame["seq"]
            idx = frame["index"]
            if idx < len(seq):
                token = seq[idx]
                frame["index"] = idx + 1
                if isinstance(token, THOL):
                    _push_thol_frame(
                        self._frames,
                        token,
                        max_materialize=self._max_materialize,
                    )
                    return THOL_SENTINEL
                return token
            else:
                cl = frame["closing"]
                frame["remaining"] -= 1
                if frame["remaining"] > 0:
                    frame["index"] = 0
                else:
                    self._frames.pop()
                if cl is not None:
                    return cl
        raise StopIteration


def _flatten_target(
    item: TARGET,
    ops: list[tuple[OpTag, Any]],
) -> None:
    ops.append((OpTag.TARGET, item))


def _flatten_wait(
    item: WAIT,
    ops: list[tuple[OpTag, Any]],
) -> None:
    steps = max(1, int(getattr(item, "steps", 1)))
    ops.append((OpTag.WAIT, steps))


def _flatten_glyph(
    item: Glyph | str,
    ops: list[tuple[OpTag, Any]],
) -> None:
    g = item.value if isinstance(item, Glyph) else str(item)
    if g not in GLYPHS_CANONICAL_SET:
        raise ValueError(f"Non-canonical glyph: {g}")
    ops.append((OpTag.GLYPH, g))


_TOKEN_DISPATCH: dict[type, Callable[[Any, list[tuple[OpTag, Any]]], None]] = {
    TARGET: _flatten_target,
    WAIT: _flatten_wait,
    Glyph: _flatten_glyph,
    str: _flatten_glyph,
}


def _flatten(
    seq: Iterable[Token] | Sequence[Token] | Any,
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
) -> list[tuple[OpTag, Any]]:
    """Return a list of operations ``(op, payload)`` where ``op`` ∈ :class:`OpTag`."""

    ops: list[tuple[OpTag, Any]] = []
    sequence = _iter_source(seq, max_materialize=max_materialize)

    def _expand(item: Any):
        if isinstance(item, THOL):
            return THOLEvaluator(item, max_materialize=max_materialize)
        return None

    for item in flatten_structure(sequence, expand=_expand):
        if item is THOL_SENTINEL:
            ops.append((OpTag.THOL, Glyph.THOL.value))
            continue
        handler = _TOKEN_DISPATCH.get(type(item))
        if handler is None:
            raise TypeError(f"Unsupported token: {item!r}")
        handler(item, ops)
    return ops
