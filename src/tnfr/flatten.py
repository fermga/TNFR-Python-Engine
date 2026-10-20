"""Admission and scope-preserving compilation of TNFR program tokens.

Compilation preserves explicit block boundaries and requested glyphs. It
neither proves a grammar word valid nor supplies a continuous law or child
target. Plain iterable grouping is sequential; explicit THOL tokens own scope.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .config.constants import GLYPHS_CANONICAL_SET
from .tokens import (
    TARGET,
    THOL,
    THOL_SENTINEL,
    WAIT,
    BlockBoundary,
    OpTag,
    Token,
    _program_count,
)
from .types import Glyph
from .utils import MAX_MATERIALIZE_DEFAULT

__all__ = ["THOLEvaluator", "parse_program_tokens"]


def _limit(value: int | None) -> int | None:
    return (
        None if value is None else _program_count(value, "max_materialize", minimum=0)
    )


class _ProgramReader:
    """One admission pass, retaining active ancestry rather than deduplicating."""

    def __init__(self, limit: int | None):
        self.limit = limit
        self.count = 0
        self.active: set[int] = set()

    def _consume(self) -> None:
        self.count += 1
        if self.limit is not None and self.count > self.limit:
            raise ValueError(f"Program exceeds max_materialize={self.limit}")

    def _enter(self, item: Any) -> None:
        if id(item) in self.active:
            raise ValueError("Cyclic program nesting is not supported")
        self.active.add(id(item))

    def sequence(self, value: Any) -> list[Token]:
        if isinstance(value, (Glyph, str, WAIT, TARGET, THOL, Mapping)):
            self._consume()
            return [self.token(value)]
        if isinstance(value, (set, frozenset)):
            raise TypeError("Program grouping must retain an explicit order")
        if isinstance(value, (bytes, bytearray)) or not isinstance(value, Iterable):
            raise TypeError("Program grouping must be an iterable of tokens")
        self._enter(value)
        try:
            result: list[Token] = []
            for item in value:
                # Leaves count themselves, including a top-level scalar token.
                # Count groups too so empty/infinite groups cannot bypass the
                # finite input budget; the outermost container is not a token.
                if not isinstance(item, (Glyph, str, WAIT, TARGET, THOL, Mapping)):
                    self._consume()
                result.extend(self.sequence(item))
            return result
        finally:
            self.active.remove(id(value))

    def token(self, item: Any) -> Token:
        if isinstance(item, (Glyph, str)):
            code = item.value if isinstance(item, Glyph) else item
            if code not in GLYPHS_CANONICAL_SET:
                raise ValueError(f"Non-canonical glyph: {code}")
            return item
        if isinstance(item, WAIT):
            return WAIT(item.steps)
        if isinstance(item, TARGET):
            if item.nodes is None:
                return TARGET(None)
            if isinstance(item.nodes, (str, bytes, bytearray, Mapping, set, frozenset)):
                raise TypeError(
                    "TARGET nodes must be an ordered collection of node identifiers"
                )
            nodes = []
            for node in item.nodes:
                self._consume()
                try:
                    hash(node)
                except TypeError as exc:
                    raise TypeError("TARGET node identifiers must be hashable") from exc
                nodes.append(node)
            return TARGET(tuple(nodes))
        self._enter(item)
        try:
            if isinstance(item, THOL):
                repeat = _program_count(item.repeat, "THOL repeat")
                close = item.force_close
                if close is not None and not isinstance(close, Glyph):
                    raise TypeError("THOL force_close must be a Glyph or None")
                return THOL(self.sequence(item.body), repeat=repeat, force_close=close)
            if not isinstance(item, Mapping) or len(item) != 1:
                raise TypeError("Program token mappings require one recognized key")
            key, value = next(iter(item.items()))
            if key == "WAIT":
                if isinstance(value, Mapping):
                    self._keys(value, {"steps"}, "WAIT")
                    value = value.get("steps", 1)
                return WAIT(value)
            if key == "TARGET":
                return self.token(TARGET(value))
            if key != "THOL":
                raise ValueError(f"Unrecognized token: {key!r}")
            if not isinstance(value, Mapping):
                raise TypeError("THOL specification must be a mapping")
            self._keys(value, {"body", "repeat", "close"}, "THOL")
            repeat = _program_count(value.get("repeat", 1), "THOL repeat")
            close = value.get("close")
            if isinstance(close, str):
                try:
                    close = Glyph(close)
                except ValueError as exc:
                    raise ValueError(f"Unknown closing glyph: {close!r}") from exc
            if close is not None and not isinstance(close, Glyph):
                raise TypeError("THOL close must be a glyph code or None")
            return THOL(
                self.sequence(value.get("body", [])),
                repeat=repeat,
                force_close=close,
            )
        finally:
            self.active.remove(id(item))

    @staticmethod
    def _keys(value: Mapping, allowed: set[str], label: str) -> None:
        if set(value) - allowed:
            raise ValueError(f"Unknown {label} fields: {set(value) - allowed!r}")


def parse_program_tokens(
    obj: Iterable[Any] | Any,
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
) -> list[Token]:
    """Validate and own a program while retaining explicit THOL bodies.

    Mapping tokens use WAIT, TARGET or THOL. Counts are positive integers;
    closure is explicit. Reusing a group twice retains both occurrences;
    cyclic ancestry is rejected. No graph or execution context is consumed.
    """

    try:
        return _ProgramReader(_limit(max_materialize)).sequence(obj)
    except RecursionError as exc:
        raise ValueError(
            "Program nesting exceeds the supported recursion depth"
        ) from exc


def _flatten(
    seq: Iterable[Token] | Any,
    *,
    max_materialize: int | None = MAX_MATERIALIZE_DEFAULT,
) -> list[tuple[OpTag, Any]]:
    """Compile admitted tokens to scoped operations, without inserting glyphs."""

    limit = _limit(max_materialize)
    tokens = parse_program_tokens(seq, max_materialize=limit)
    ops: list[tuple[OpTag, Any]] = []

    def emit(tag: OpTag, payload: Any) -> None:
        if limit is not None and len(ops) >= limit:
            raise ValueError(f"Expanded program exceeds max_materialize={limit}")
        ops.append((tag, payload))

    def compile_items(items: list[Token], path: tuple[int, ...]) -> None:
        for index, item in enumerate(items):
            item_path = (*path, index)
            if isinstance(item, THOL):
                for iteration in range(item.repeat):
                    boundary = BlockBoundary(item_path, iteration, item.repeat)
                    emit(OpTag.BLOCK_ENTER, boundary)
                    if iteration == 0:
                        emit(OpTag.THOL, Glyph.THOL.value)
                    compile_items(item.body, (*item_path, iteration))
                    if item.force_close is not None:
                        emit(OpTag.GLYPH, item.force_close.value)
                    emit(OpTag.BLOCK_EXIT, boundary)
            elif isinstance(item, TARGET):
                emit(OpTag.TARGET, item)
            elif isinstance(item, WAIT):
                emit(OpTag.WAIT, item.steps)
            else:
                emit(OpTag.GLYPH, item.value if isinstance(item, Glyph) else item)

    try:
        compile_items(tokens, ())
    except RecursionError as exc:
        raise ValueError(
            "Program nesting exceeds the supported recursion depth"
        ) from exc
    return ops


class THOLEvaluator:
    """Compatibility token view of one admitted expanded block.

    This view omits scope metadata; execution uses the scoped compiler instead.
    A closure appears only when explicitly supplied in force_close.
    """

    def __init__(self, item: THOL, *, max_materialize=MAX_MATERIALIZE_DEFAULT):
        if not isinstance(item, THOL):
            raise TypeError("THOLEvaluator requires a THOL token")
        self._operations = iter(_flatten([item], max_materialize=max_materialize))

    def __iter__(self) -> THOLEvaluator:
        return self

    def __next__(self) -> Token | object:
        for tag, payload in self._operations:
            if tag is OpTag.THOL:
                return THOL_SENTINEL
            if tag is OpTag.WAIT:
                return WAIT(payload)
            if tag in {OpTag.GLYPH, OpTag.TARGET}:
                return payload
        raise StopIteration
