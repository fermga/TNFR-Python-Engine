"""Token primitives for the TNFR DSL."""

from __future__ import annotations

from .compat.dataclass import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Optional, Sequence, Union

from .types import Glyph, Node


@dataclass(slots=True)
class WAIT:
    """Wait a number of steps without applying glyphs."""

    steps: int = 1


@dataclass(slots=True)
class TARGET:
    """Select the subset of nodes for subsequent glyphs."""

    nodes: Optional[Iterable[Node]] = None  # ``None`` targets all nodes


@dataclass(slots=True)
class THOL:
    """THOL block that opens self-organisation."""

    body: Sequence[Any]
    repeat: int = 1
    force_close: Optional[Glyph] = None


Token = Union[Glyph, WAIT, TARGET, THOL, str]

# Sentinel used internally to mark the boundaries of a THOL block during flattening
THOL_SENTINEL = object()


class OpTag(Enum):
    """Operation tags emitted by the flattening step."""

    TARGET = auto()
    WAIT = auto()
    GLYPH = auto()
    THOL = auto()


__all__ = [
    "Node",
    "WAIT",
    "TARGET",
    "THOL",
    "Token",
    "THOL_SENTINEL",
    "OpTag",
]
