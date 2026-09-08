"""Mutation (ZHIR) operator.

Purpose: controlled phase transformation (theta -> theta').
Physics: trigger when dEPI/dt > xi; identity preserved (epi_kind).
Grammar: U4b requires prior IL + recent destabilizer (OZ/VAL).
Effects: regime shift through theta; keeps EPI, vf, and identity stable.
Preconditions: active vf; signed sampled velocity>xi; two EPI samples.
An explicit ZHIR_MIN_VF may further tighten the active-capacity requirement.
Typical: IL->OZ->ZHIR->IL; THOL->OZ->ZHIR; IL->VAL->ZHIR->IL.
Avoid: ZHIR->ZHIR; AL->ZHIR; ZHIR->OZ; OZ->ZHIR->OZ.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import MUTATION
from ..constants.aliases import ALIAS_EPI_KIND, ALIAS_THETA
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Mutation(Operator):
    """Controlled phase transform; regime shift with identity preserved.

    Invariants: maintain EPI and epi_kind; theta shifts; vf stable; dnfr
    elevated before the transformation.
    The phase proposal is RNG-free and depends only on the validated state and
    configuration; it does not consume or depend on a random seed.
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
        from ._mutation_stage_kernel import propose_mutation_network_stage
        from .factor_contracts import resolve_runtime_operator_factors

        factors = resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        proposal = propose_mutation_network_stage(
            G,
            node,
            factors,
            tau=kw.get("tau"),
        )

        validate_postconditions = kw.get(
            "validate_postconditions", False
        ) or G.graph.get("VALIDATE_OPERATOR_POSTCONDITIONS", False)
        state_before = {
            "theta": proposal.phase.theta_before,
            "epi_kind": proposal.epi_kind_before,
        }

        execution_kwargs = dict(kw)
        execution_kwargs["_mutation_stage_proposal"] = proposal
        super()._execute(G, node, **execution_kwargs)

        # Verify postconditions if enabled
        if validate_postconditions:
            self._verify_postconditions(G, node, state_before)
        from ._mutation_stage_kernel import emit_mutation_lifecycle_log

        emit_mutation_lifecycle_log(proposal)

    def _after_glyph_application(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        """Merge proposal-bound evidence after phase and history commit."""

        from ._mutation_stage_kernel import (
            MutationNetworkStageProposal,
            commit_mutation_lifecycle,
        )

        proposal = kw.get("_mutation_stage_proposal")
        if not isinstance(proposal, MutationNetworkStageProposal):
            raise RuntimeError("Mutation lifecycle proposal is missing")
        if proposal.node != node:
            raise RuntimeError("Mutation lifecycle proposal target changed")
        theta_after = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
        if theta_after != proposal.phase.theta_after:
            raise RuntimeError("Mutation phase result diverged from its proposal")
        for key, value in proposal.phase.telemetry_items:
            if G.nodes[node].get(key) != value:
                raise RuntimeError(
                    "Mutation glyph telemetry diverged from its proposal"
                )
        commit_mutation_lifecycle(G, proposal)

    def _compute_epi_acceleration(self, G: TNFRGraph, node: Any) -> float:
        """Return the shared structural-acceleration magnitude without writes."""

        from .nodal_equation import compute_d2epi_dt2

        return abs(compute_d2epi_dt2(G, node, store=False))

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
