"""Mutation (ZHIR) operator.

Purpose: controlled phase transformation (theta -> theta').
Physics: trigger when dEPI/dt > xi; identity preserved (epi_kind).
Grammar: U4b requires prior IL + recent destabilizer (OZ/VAL).
Effects: regime shift; may adjust epi; keeps vf and identity stable.
Preconditions: active vf; signed sampled velocity>xi; two EPI samples.
An explicit ZHIR_MIN_VF may further tighten the active-capacity requirement.
Typical: IL->OZ->ZHIR->IL; THOL->OZ->ZHIR; IL->VAL->ZHIR->IL.
Avoid: ZHIR->ZHIR; AL->ZHIR; ZHIR->OZ; OZ->ZHIR->OZ.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import MUTATION
from ..constants.aliases import ALIAS_EPI_KIND
from ..types import Glyph, TNFRGraph
from ._argument_validation import finite_real, require_list_sink
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

    def _capture_state(self, G: TNFRGraph, node: Any) -> dict[str, Any]:
        """Capture scalar channels plus the canonical structural identity."""

        state = super()._capture_state(G, node)
        state["epi_kind"] = get_attr(
            G.nodes[node],
            ALIAS_EPI_KIND,
            None,
            strict=True,
            conv=lambda value: None if value is None else str(value),
        )
        return state

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply ZHIR; detect bifurcation; optional post checks."""
        # Capture state before mutation for postcondition verification
        validate_postconditions = kw.get(
            "validate_postconditions", False
        ) or G.graph.get("VALIDATE_OPERATOR_POSTCONDITIONS", False)

        state_before = None
        if validate_postconditions:
            state_before = self._capture_state(G, node)

        # Compute structural acceleration before base operator
        d2_epi = self._compute_epi_acceleration(G, node)

        # Resolve and validate the active acceleration threshold before the
        # base glyph can change phase, history, provenance, or telemetry.
        tau_raw = kw.get("tau")
        if tau_raw is None:
            tau_raw = G.graph.get(
                "BIFURCATION_THRESHOLD_TAU",
                G.graph.get("ZHIR_BIFURCATION_THRESHOLD", 0.5),
            )
        tau = finite_real(
            tau_raw,
            operator=self.name,
            label="tau",
            lower=0.0,
        )
        if d2_epi > tau:
            require_list_sink(
                G.graph, "zhir_bifurcation_events", operator=self.name
            )

        # Apply base operator (glyph, preconditions, metrics)
        super()._execute(G, node, **kw)

        # Detect bifurcation potential if acceleration exceeds threshold
        if d2_epi > tau:
            self._detect_bifurcation_potential(G, node, d2_epi=d2_epi, tau=tau)

        # Verify postconditions if enabled
        if validate_postconditions and state_before is not None:
            self._verify_postconditions(G, node, state_before)

    def _compute_epi_acceleration(self, G: TNFRGraph, node: Any) -> float:
        """Return the shared structural-acceleration magnitude without writes."""

        from .nodal_equation import compute_d2epi_dt2

        return abs(compute_d2epi_dt2(G, node, store=False))

    def _detect_bifurcation_potential(
        self, G: TNFRGraph, node: Any, d2_epi: float, tau: float
    ) -> None:
        """Flag bifurcation potential (d2_epi>tau) and log event."""
        import logging

        logger = logging.getLogger(__name__)

        # set telemetry flags for grammar validation
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

    def _validate_hard_invariants(self, G: TNFRGraph, node: Any) -> None:
        """Require ZHIR's signed positive-growth trigger on every application."""
        from ._mutation_gate import validate_mutation_runtime_gate

        validate_mutation_runtime_gate(G.nodes[node], G.graph)

    def _verify_postconditions(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> None:
        """Verify phase shift, identity preserved, bifurcation handled."""
        from .postconditions.mutation import (
            verify_bifurcation_handled,
            verify_identity_preserved,
            verify_phase_transformed,
        )

        # Verify phase transformation
        verify_phase_transformed(G, node, state_before["theta"])

        # Verify structural identity independently from glyph provenance.
        verify_identity_preserved(G, node, state_before.get("epi_kind"))

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
            epi_kind_before=state_before.get("epi_kind"),
        )
