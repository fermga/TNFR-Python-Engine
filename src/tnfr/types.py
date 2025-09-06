"""Type definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


@dataclass(slots=True)
class NodeState:
    EPI: float = 0.0
    vf: float = 1.0  # νf
    theta: float = 0.0  # θ
    Si: float = 0.5
    epi_kind: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class Glyph(str, Enum):
    """Canonical TNFR glyphs."""

    AL = "AL"
    EN = "EN"
    IL = "IL"
    OZ = "OZ"
    UM = "UM"
    RA = "RA"
    SHA = "SHA"
    VAL = "VAL"
    NUL = "NUL"
    THOL = "THOL"
    ZHIR = "ZHIR"
    NAV = "NAV"
    REMESH = "REMESH"
