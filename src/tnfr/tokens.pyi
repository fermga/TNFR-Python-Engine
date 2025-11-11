from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Union

from ._compat import TypeAlias
from .types import Glyph, NodeId

__all__: tuple[str, ...]

Node: TypeAlias = NodeId

@dataclass
class WAIT:
    steps: int = 1

@dataclass
class TARGET:
    nodes: Optional[Iterable[Node]] = None

@dataclass
class THOL:
    body: Sequence["Token"]
    repeat: int = 1
    force_close: Optional[Glyph] = None

Token: TypeAlias = Union[Glyph, WAIT, TARGET, THOL, str]

THOL_SENTINEL: object

class OpTag(Enum):
    TARGET = ...
    WAIT = ...
    GLYPH = ...
    THOL = ...
