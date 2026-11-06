"""Expansion (VAL) operator.

Purpose: apply the configured capacity and optional form scale.
Effect: raises vf; edge-aware mode also scales/projects EPI.
Stored pressure, phase and support are unchanged. Grammar: destabilizer (U2).
Optional strict thresholds and U2 admission do not prove convergence.
Typical: VAL -> IL (stabilize); OZ -> VAL (dissonance then expand).
Avoid: repeated VAL without stabilization; immediate VAL -> NUL.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import EXPANSION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Expansion(Operator):
    """Scale capacity and, in edge-aware mode, the admitted scalar form.

    The default edge-aware branch respects configured EPI bounds. Disabling
    it preserves EPI. Neither branch directly changes pressure or support;
    any subsequent pressure refresh is a separate operation. Multiplying
    capacity does not create state dimensions or an autonomous growth law.
    Grammar classifies VAL as a destabilizer under the U2 policy.

    Example:
      expand then stabilize.
      >>> from tnfr.structural import create_nfr, run_sequence
      >>> from tnfr.operators.definitions import Expansion, Coherence
      >>> G, node = create_nfr("theta", epi=0.47, vf=0.95)
      >>> run_sequence(G, node, [Expansion(), Coherence()])

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
