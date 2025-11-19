"""Recursivity (REMESH) operator.

Purpose: propagate fractal pattern echoes across nested EPIs.
Physics: epi(t) references epi(t-τ); multi-scale identity retention.
Grammar: generator/closure; depth>1 enforces U5 stabilizers nearby.
Telemetry: depth, epi_before, vf_before for recursion analysis.
Typical: THOL->REMESH, REMESH->IL, VAL->REMESH, REMESH->RA.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import RECURSIVITY
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
 

class Recursivity(Operator):
    """Propagate fractal echoes; enforce multi-scale identity retention.

    depth>1: requires nearby IL/THOL (U5 coherence). Metrics minimal.
    """

    __slots__ = ("depth",)
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH

    def __init__(self, depth: int = 1):
        """Set recursion depth (>=1)."""
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Run REMESH precondition validator."""
        from .preconditions import validate_recursivity

        validate_recursivity(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect REMESH metrics (epi,vf before)."""
        from .metrics import recursivity_metrics

        return recursivity_metrics(
            G, node, state_before["epi"], state_before["vf"]
        )
