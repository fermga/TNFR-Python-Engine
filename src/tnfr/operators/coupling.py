"""Coupling (UM) operator.

Purpose: synchronize phases; establish structural links; enable resonance.
Physics: aligns theta across neighbors; may reduce dnfr locally.
Grammar: coupling requires phase compatibility (U3) when strict enabled.
Effects: phase spread narrows; vf may align; epi untouched.
Preconditions: active epi & vf; optional phase window; connectivity.
Typical: AL->UM; UM->RA; UM->IL; EN->UM; UM->THOL.
Avoid: UM with insufficient epi/vf or extreme phase mismatch.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import COUPLING
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Coupling(Operator):
    """Synchronize phases; create/strengthen links; enable resonance.

    Invariants: preserves epi; adjusts theta; may align vf; can lower dnfr.
    Config: UM_MIN_EPI, UM_MIN_VF, UM_STRICT_PHASE_CHECK, UM_MAX_PHASE_DIFF.
    Common: AL->UM, UM->RA, UM->IL, EN->UM, UM->THOL.
    """

    __slots__ = ()
    name: ClassVar[str] = COUPLING
    glyph: ClassVar[Glyph] = Glyph.UM

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate UM-specific preconditions."""
        from .preconditions import validate_coupling

        validate_coupling(G, node)

    def _capture_state(self, G: TNFRGraph, node: Any) -> dict[str, Any]:
        """Capture node state before apply; include edge count."""
        # Get base state (epi, vf, dnfr, theta)
        state = super()._capture_state(G, node)

        # Add edge count for coupling-specific metrics
        state["edges"] = G.degree(node)

        return state

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect UM-specific metrics with expanded canonical measurements."""
        from .metrics import coupling_metrics

        return coupling_metrics(
            G,
            node,
            state_before["theta"],
            dnfr_before=state_before["dnfr"],
            vf_before=state_before["vf"],
            edges_before=state_before.get("edges", None),
            epi_before=state_before["epi"],
        )
