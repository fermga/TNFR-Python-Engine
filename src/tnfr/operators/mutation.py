"""Mutation (ZHIR) operator.

Purpose: controlled phase transformation (theta -> theta').
Physics: trigger when dEPI/dt > xi; identity preserved (epi_kind).
Grammar: U4b requires prior IL + recent destabilizer (OZ/VAL).
Effects: regime shift; may adjust epi; keeps vf and identity stable.
Preconditions: vf>=ZHIR_MIN_VF; velocity>xi; history length; coupling.
Typical: IL->OZ->ZHIR->IL; THOL->OZ->ZHIR; IL->VAL->ZHIR->IL.
Avoid: ZHIR->ZHIR; AL->ZHIR; ZHIR->OZ; OZ->ZHIR->OZ.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import MUTATION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Mutation(Operator):
    """Controlled phase transform; regime shift with identity preserved.

    Invariants: maintain epi_kind; theta shifts; vf stable; dnfr elevated pre.
    Grammar: needs prior IL + recent OZ/VAL (U4b); may flag bifurcation.
    Typical: IL->OZ->ZHIR->IL; IL->VAL->ZHIR->IL; OZ->ZHIR->THOL.
    Avoid: ZHIR->ZHIR; AL->ZHIR; ZHIR->OZ; OZ->ZHIR->OZ.
    Metrics: theta_shift, regime_changed, depi_dt, threshold_met, d2_epi.
    """

    __slots__ = ()
    name: ClassVar[str] = MUTATION
    glyph: ClassVar[Glyph] = Glyph.ZHIR

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply ZHIR; detect bifurcation; optional post checks."""
        # Capture state before mutation for postcondition verification
        validate_postconditions = (
            kw.get("validate_postconditions", False)
            or G.graph.get("VALIDATE_OPERATOR_POSTCONDITIONS", False)
        )

        state_before = None
        if validate_postconditions:
            state_before = self._capture_state(G, node)
            # Also capture epi_kind if tracked
            state_before["epi_kind"] = G.nodes[node].get("epi_kind")

        # Compute structural acceleration before base operator
        d2_epi = self._compute_epi_acceleration(G, node)

        # Get bifurcation threshold (tau) from kwargs or graph config
        tau = kw.get("tau")
        if tau is None:
            # Resolve tau: canonical, operator-specific, fallback
            tau = float(
                G.graph.get(
                    "BIFURCATION_THRESHOLD_TAU",
                    G.graph.get("ZHIR_BIFURCATION_THRESHOLD", 0.5),
                )
            )

        # Apply base operator (glyph, preconditions, metrics)
        super().__call__(G, node, **kw)

        # Detect bifurcation potential if acceleration exceeds threshold
        if d2_epi > tau:
            self._detect_bifurcation_potential(G, node, d2_epi=d2_epi, tau=tau)

        # Verify postconditions if enabled
        if validate_postconditions and state_before is not None:
            self._verify_postconditions(G, node, state_before)

    def _compute_epi_acceleration(self, G: TNFRGraph, node: Any) -> float:
        """Finite diff second derivative of epi history; abs value."""

        # Get EPI history (maintained by node for temporal analysis)
        history = G.nodes[node].get("epi_history", [])

        # Need at least 3 points for second derivative
        if len(history) < 3:
            return 0.0

        # Finite difference: d²EPI/dt² ≈ (EPI_t - 2*EPI_{t-1} + EPI_{t-2})
        epi_t = float(history[-1])
        epi_t1 = float(history[-2])
        epi_t2 = float(history[-3])

        d2_epi = epi_t - 2.0 * epi_t1 + epi_t2

        return abs(d2_epi)

    def _detect_bifurcation_potential(
        self, G: TNFRGraph, node: Any, d2_epi: float, tau: float
    ) -> None:
        """Flag bifurcation potential (d2_epi>tau) and log event."""
        import logging

        logger = logging.getLogger(__name__)

        # Set telemetry flags for grammar validation
        G.nodes[node]["_zhir_bifurcation_potential"] = True
        G.nodes[node]["_zhir_d2epi"] = d2_epi
        G.nodes[node]["_zhir_tau"] = tau

        # Record bifurcation detection event in graph for analysis
        bifurcation_events = G.graph.setdefault("zhir_bifurcation_events", [])
        bifurcation_events.append(
            {
                "node": node,
                "d2_epi": d2_epi,
                "tau": tau,
                "timestamp": len(G.nodes[node].get("glyph_history", [])),
            }
        )

        # Log informative message
        logger.info(
            f"Node {node}: ZHIR bifurcation potential detected "
            f"(∂²EPI/∂t²={d2_epi:.3f} > τ={tau}). "
            "Consider THOL for bifurcation or IL for stabilization."
        )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate ZHIR-specific preconditions."""
        from .preconditions import validate_mutation

        validate_mutation(G, node)

    def _verify_postconditions(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> None:
        """Verify phase shift, identity preserved, bifurcation handled."""
        from .postconditions.mutation import (
            verify_phase_transformed,
            verify_identity_preserved,
            verify_bifurcation_handled,
        )

        # Verify phase transformation
        verify_phase_transformed(G, node, state_before["theta"])

        # Verify identity preservation (if tracked)
        epi_kind_before = state_before.get("epi_kind")
        if epi_kind_before is not None:
            verify_identity_preserved(G, node, epi_kind_before)

        # Verify bifurcation handling
        verify_bifurcation_handled(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect ZHIR-specific metrics."""
        from .metrics import mutation_metrics

        return mutation_metrics(
            G,
            node,
            state_before["theta"],
            state_before["epi"],
            vf_before=state_before.get("vf"),
            dnfr_before=state_before.get("dnfr"),
        )
