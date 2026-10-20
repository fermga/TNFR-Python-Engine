from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, TypeAlias, Union

from .types import Glyph, NodeId

__all__: list[str]

Node: TypeAlias = NodeId

@dataclass
class WAIT:
    steps: int = 1

@dataclass
class TARGET:
    nodes: Optional[Iterable[Node]] = None

@dataclass
class THOL:
    body: Sequence[Any]
    repeat: int = 1
    force_close: Optional[Glyph] = None

Token: TypeAlias = Union[Glyph, WAIT, TARGET, THOL, str]

THOL_SENTINEL: object

@dataclass(frozen=True)
class BlockBoundary:
    path: tuple[int, ...]
    iteration: int
    repeat: int

class OpTag(Enum):
    TARGET = ...
    WAIT = ...
    GLYPH = ...
    THOL = ...
    BLOCK_ENTER = ...
    BLOCK_EXIT = ...
