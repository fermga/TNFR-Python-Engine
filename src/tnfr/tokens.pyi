from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence

from .types import Glyph, NodeId
from ._compat import TypeAlias

__all__: tuple[str, ...]

Node: TypeAlias = NodeId


@dataclass(slots=True)
class WAIT:
    steps: int = 1


@dataclass(slots=True)
class TARGET:
    nodes: Optional[Iterable[Node]] = None


@dataclass(slots=True)
class THOL:
    body: Sequence["Token"]
    repeat: int = 1
    force_close: Optional[Glyph] = None


Token: TypeAlias = Glyph | WAIT | TARGET | THOL | str

THOL_SENTINEL: object


class OpTag(Enum):
    TARGET = ...
    WAIT = ...
    GLYPH = ...
    THOL = ...
