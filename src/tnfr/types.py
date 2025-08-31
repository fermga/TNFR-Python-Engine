from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


@dataclass
class NodeState:
    EPI: float = 0.0
    vf: float = 1.0   # νf
    theta: float = 0.0  # θ
    Si: float = 0.5
    epi_kind: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> Dict[str, Any]:
        d = {"EPI": self.EPI, "νf": self.vf, "θ": self.theta, "Si": self.Si, "EPI_kind": self.epi_kind}
        d.update(self.extra)
        return d


class Glyph(str, Enum):
    """Glifos canónicos de la TNFR."""

    AL = "A’L"
    EN = "E’N"
    IL = "I’L"
    OZ = "O’Z"
    UM = "U’M"
    RA = "R’A"
    SHA = "SH’A"
    VAL = "VA’L"
    NUL = "NU’L"
    THOL = "T’HOL"
    ZHIR = "Z’HIR"
    NAV = "NA’V"
    REMESH = "RE’MESH"

