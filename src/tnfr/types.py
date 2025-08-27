from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class NodeState:
    EPI: float = 0.0
    vf: float = 1.0   # νf
    theta: float = 0.0  # θ
    Si: float = 0.5
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> Dict[str, Any]:
        d = {"EPI": self.EPI, "νf": self.vf, "θ": self.theta, "Si": self.Si}
        d.update(self.extra)
        return d
