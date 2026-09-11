"""Expansion (VAL) operator.

Purpose: increase structural degrees of freedom for exploration.
Physics: raises EPI + vf; may widen ΔNFR. Grammar: destabilizer (U2).
Effects: form scales, pressure can rise, phase unchanged.
Preconditions: bounded ΔNFR; follow with IL/THOL for convergence.
Typical: VAL -> IL (stabilize); OZ -> VAL (dissonance then expand).
Avoid: repeated VAL without stabilization; immediate VAL -> NUL.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import EXPANSION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Expansion(Operator):
    """Expand structural scope for exploration.

    Raises EPI and vf; may elevate delta NFR (needs later IL/THOL).
    Grammar: destabilizer (U2) so stabilizers must follow.

    Example:
      expand then stabilize.
      >>> from tnfr.structural import create_nfr, run_sequence
      >>> from tnfr.operators.definitions import Expansion, Coherence
      >>> G, node = create_nfr("theta", epi=0.47, vf=0.95)
      >>> run_sequence(G, node, [Expansion(), Coherence()])

    Domains: biomedical growth; cognitive broadening; social scaling.
    """

    __slots__ = ()
    name: ClassVar[str] = EXPANSION
    glyph: ClassVar[Glyph] = Glyph.VAL

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate VAL-specific preconditions."""
        from .preconditions import validate_expansion

        validate_expansion(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect VAL-specific metrics."""
        from .metrics import expansion_metrics

        return expansion_metrics(
            G,
            node,
            state_before["vf"],
            state_before["epi"],
        )
