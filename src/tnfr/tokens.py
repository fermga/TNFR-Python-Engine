"""Token primitives for the TNFR DSL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from operator import index as integer_index
from typing import Any, Iterable, Sequence

import numpy as np

from .types import Glyph, Node


def _program_count(value: Any, label: str, *, minimum: int = 1) -> int:
    """Admit a discrete count without truncation or Boolean coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer, not boolean")
    try:
        count = integer_index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be an integer") from exc
    if count < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return count


@dataclass(slots=True)
class WAIT:
    """Request extra step-function calls, without an explicit program glyph.

    A supplied step function can itself select operators or evolve state.
    This count does not declare a physical duration or event-free interval.
    """

    steps: int = 1

    def __post_init__(self) -> None:
        self.steps = _program_count(self.steps, "WAIT steps")


@dataclass(slots=True)
class TARGET:
    """Select the subset of nodes for subsequent glyphs."""

    nodes: Iterable[Node] | None = None  # ``None`` targets all nodes


@dataclass(slots=True)
class THOL:
    """Sequential program block with lexical targets and explicit closure.

    Grouping does not itself admit a collective NFR, child target or closed
    dynamics. The THOL opening executes once; the body and an explicitly
    requested final glyph execute ``repeat`` times. ``None`` adds no glyph.
    """

    body: Sequence[Any]
    repeat: int = 1
    force_close: Glyph | None = None

    def __post_init__(self) -> None:
        self.repeat = _program_count(self.repeat, "THOL repeat")
        if self.force_close is not None and not isinstance(self.force_close, Glyph):
            raise TypeError("THOL force_close must be a Glyph or None")


@dataclass(frozen=True, slots=True)
class BlockBoundary:
    """Program scope metadata, never a canonical structural operator.

    ``path`` identifies the source block; zero-based ``iteration`` identifies
    one body invocation. Nested occurrences retain their enclosing scope.
    """

    path: tuple[int, ...]
    iteration: int
    repeat: int


Token = Glyph | WAIT | TARGET | THOL | str

# Compatibility sentinel for a THOL opening in THOLEvaluator's token view.
THOL_SENTINEL = object()


class OpTag(Enum):
    """Operation tags emitted by the flattening step."""

    TARGET = auto()
    WAIT = auto()
    GLYPH = auto()
    THOL = auto()
    BLOCK_ENTER = auto()
    BLOCK_EXIT = auto()


__all__ = [
    "Node",
    "WAIT",
    "TARGET",
    "THOL",
    "Token",
    "THOL_SENTINEL",
    "BlockBoundary",
    "OpTag",
]
