"""Resonance (RA) operator.

Purpose: propagate coherence through coupled phase-aligned nodes.
Physics: circulates structural pattern without altering epi identity.
Grammar: requires prior coupling (U3) when strict phase validation active.
Effects: raises global C(t); may amplify vf; keeps epi form intact.
Preconditions: coherent epi, edges, phase alignment, sufficient vf.
Typical: UM->RA; IL->RA; AL->RA; RA->IL; RA->EN.
Avoid: OZ->RA without IL; SHA->RA; repeated RA chains without IL.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import RESONANCE
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Resonance(Operator):
    """Propagate coherence across coupled nodes; amplify network alignment.

    Invariants: preserve epi; adjust vf moderately; may raise C(t); limit dnfr.
    Typical: UM->RA, IL->RA, AL->RA, RA->IL, RA->EN. Avoid OZ->RA w/o IL.
    Metrics: propagation distance, amplification factor, phase order param.
    """

    __slots__ = ()
    name: ClassVar[str] = RESONANCE
    glyph: ClassVar[Glyph] = Glyph.RA

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate RA-specific preconditions."""
        from .preconditions import validate_resonance

        validate_resonance(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect RA metrics including vf amplification tracking."""
        from .metrics import resonance_metrics

        return resonance_metrics(
            G,
            node,
            state_before["epi"],
            vf_before=state_before["vf"],  # vf for amplification tracking
        )
