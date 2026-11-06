"""Coupling (UM) operator.

Purpose: synchronize phases; establish structural links; enable resonance.
Physics: aligns theta across neighbors; may reduce dnfr locally.
Grammar: coupling always requires phase compatibility (U3).
Effects: configured circular phase alignment; vf may align; epi untouched.
Functional-link proposals may add compatible nonneighbors; invocation is supplied.
An explicit request may connect a target to a supplied isolated NFR.
Preconditions: selected epi & vf activity policy; mandatory phase/domain gates.
Typical: AL->UM; UM->RA; UM->IL; EN->UM; UM->THOL.
Avoid: UM with insufficient epi/vf or extreme phase mismatch.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import COUPLING
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Coupling(Operator):
    """Apply circular phase alignment and optional new functional links.

    Invariants: preserves epi; adjusts theta; may align vf; can lower dnfr.
    Config: UM_MIN_EPI, UM_MIN_VF, UM_STRICT_PHASE_CHECK, UM_MAX_PHASE_DIFF.
    Common: AL->UM, UM->RA, UM->IL, EN->UM, UM->THOL.
    """

    __slots__ = ()
    name: ClassVar[str] = COUPLING
    glyph: ClassVar[Glyph] = Glyph.UM

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply UM, or atomically connect a supplied partner.

        ``potential_partner`` names a supplied, distinct isolated NFR. The
        target may have existing support; compatible outgoing neighbors and
        the mandatory partner share the existing UM map. Participants must
        pass initial/final U3 and the new link its configured affinity.
        Its provenance is the explicit request, not an autonomous selection.
        Omitting the partner preserves the ordinary existing-neighbor action.
        ``potential_scope='disjoint_components'`` instead admits two distinct
        nontrivial components of an undirected non-multigraph and protects
        every relation incident to the declared phase writers. Only this
        target records UM; secondary participants retain their own contexts.
        ``preserve_relations=True`` instead protects existing relations during
        ordinary UM through the same atomic stage. It requires fixed support,
        disabled functional links and an undirected non-multigraph.
        """
        from ._argument_validation import strict_bool
        from ._coupling_potential import _resolve_potential_scope

        preserve_relations = strict_bool(
            kw.pop("preserve_relations", False),
            operator="Coupling",
            label="preserve_relations",
        )
        _resolve_potential_scope(
            kw.get("potential_scope", "isolated"), kw.get("potential_partner")
        )
        if kw.get("potential_partner") is None and not preserve_relations:
            super().__call__(G, node, **kw)
            return
        from .network_stage import execute_coupling_stage

        if preserve_relations:
            kw["preserve_relations"] = True
        execute_coupling_stage(G, self, (node,), **kw)

    def _validate_requested_hard_invariants(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        partner = kw.get("potential_partner")
        from ._coupling_potential import _resolve_potential_scope

        scope = _resolve_potential_scope(kw.get("potential_scope", "isolated"), partner)
        if partner is None:
            self._validate_hard_invariants(G, node)
            return
        from .preconditions import validate_phase_gate_u3

        validate_phase_gate_u3(
            G, node, "Coupling", potential_partner=partner, potential_scope=scope
        )

    def _validate_requested_preconditions(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        partner = kw.get("potential_partner")
        from ._coupling_potential import _resolve_potential_scope

        scope = _resolve_potential_scope(kw.get("potential_scope", "isolated"), partner)
        if partner is None:
            self._validate_preconditions(G, node)
            return
        from .preconditions import validate_coupling

        validate_coupling(G, node, potential_partner=partner, potential_scope=scope)

    def _validate_application_preconditions(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        """Reject invalid branch controls before grammar or monitor writes."""
        from .factor_contracts import _resolve_coupling_flags

        _resolve_coupling_flags(G.graph)
        super()._validate_application_preconditions(G, node, **kw)

    def _validate_hard_invariants(self, G: TNFRGraph, node: Any) -> None:
        """U3 phase gate (Invariant #2) — always enforced, cannot be disabled."""
        from .preconditions import validate_phase_gate_u3

        validate_phase_gate_u3(G, node, "Coupling")

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


_CANONICAL_COUPLING_CLASS = Coupling
_CANONICAL_COUPLING_CALL = Coupling.__call__
_CANONICAL_COUPLING_EXECUTE = Coupling._execute
