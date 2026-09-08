"""Coherence (IL) operator.

Purpose: stabilize form; reduce delta NFR; raise coherence.
Physics: negative feedback drives delta NFR toward 0.
Grammar: stabilizer (U2); safe closure component.
Effects: pressure lowers; EPI preserved; optional phase locking.
Preconditions: active structure; prior destabilizer if recent IL already.
Typical: AL->UM->IL; OZ->IL; VAL->IL. Avoid redundant IL chains.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from ..alias import get_attr, set_attr
from ..config.operator_names import COHERENCE
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..types import Glyph, TNFRGraph
from ._argument_validation import (
    finite_node_real,
    finite_real,
    nonnegative_integer,
    reject_operator_argument,
    require_list_sink,
    strict_bool,
)
from ._coherence_stage_kernel import (
    CoherencePhaseProposal,
    CoherenceStageProposal,
    coherence_reduction_event,
    coherence_tracking_event,
    propose_coherence_phase,
    propose_coherence_stage,
)
from .definitions_base import Operator
from .factor_contracts import resolve_runtime_operator_factors

class Coherence(Operator):
    """Stabilize alignment; compress delta NFR; boost coherence.

    Drives local equilibrium (delta NFR down). Often follows
    Emission/Coupling or contains OZ/VAL effects. Phase locking optional.
    """

    __slots__ = ()
    name: ClassVar[str] = COHERENCE
    glyph: ClassVar[Glyph] = Glyph.IL

    def _validate_application_preconditions(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        """Validate every active public IL input before grammar selection."""

        nonnegative_integer(
            kw.get("coherence_radius", 1),
            operator=self.name,
            label="coherence_radius",
        )
        finite_real(
            kw.get("phase_locking_coefficient", 0.3),
            operator=self.name,
            label="phase_locking_coefficient",
            lower=0.0,
            upper=1.0,
        )
        self._validate_state(G, node)
        self._validate_sinks_and_monitor(G, kw)

        validation_enabled = bool(kw.get("validate_preconditions", True)) and bool(
            G.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        )
        if validation_enabled:
            self._validate_precondition_config(G)
            from .preconditions.coherence import validate_coherence_strict

            validate_coherence_strict(
                G,
                node,
                emit_warnings=not bool(
                    kw.get("_defer_il_precondition_warnings", False)
                ),
            )

        base_kwargs = dict(kw)
        base_kwargs.pop("_defer_il_precondition_warnings", None)
        if validation_enabled:
            # The IL validator above owns warning deferral for network preflight.
            base_kwargs["validate_preconditions"] = False
        super()._validate_application_preconditions(G, node, **base_kwargs)

    def _validate_state(self, G: TNFRGraph, node: Any) -> None:
        """Require a finite scalar IL state with nonnegative capacity."""

        finite_real(
            get_attr(G.nodes[node], ALIAS_EPI, 0.0, strict=True),
            operator=self.name,
            label="EPI state",
        )
        finite_node_real(
            G.nodes[node],
            ALIAS_VF,
            0.0,
            operator=self.name,
            label="nu_f state",
            lower=0.0,
        )
        finite_node_real(
            G.nodes[node],
            ALIAS_DNFR,
            0.0,
            operator=self.name,
            label="DeltaNFR state",
        )
        finite_node_real(
            G.nodes[node],
            ALIAS_THETA,
            0.0,
            operator=self.name,
            label="theta state",
        )

    def _validate_precondition_config(self, G: TNFRGraph) -> None:
        """Validate the optional strict IL thresholds before their consumer."""

        from ..config.thresholds import (
            DNFR_IL_CRITICAL,
            EPI_IL_MIN,
            VF_IL_MIN,
        )

        raw = G.graph.get("IL_PRECONDITIONS", {})
        if not isinstance(raw, Mapping):
            reject_operator_argument(
                self.name, "IL_PRECONDITIONS must be a mapping"
            )
        finite_real(
            raw.get("min_epi", EPI_IL_MIN),
            operator=self.name,
            label="IL_PRECONDITIONS.min_epi",
        )
        finite_real(
            raw.get("min_vf", VF_IL_MIN),
            operator=self.name,
            label="IL_PRECONDITIONS.min_vf",
            lower=0.0,
        )
        finite_real(
            raw.get("dnfr_critical_threshold", DNFR_IL_CRITICAL),
            operator=self.name,
            label="IL_PRECONDITIONS.dnfr_critical_threshold",
            lower=0.0,
        )
        for key in ("warn_isolated", "warn_zero_dnfr"):
            if key in raw:
                strict_bool(
                    raw[key],
                    operator=self.name,
                    label=f"IL_PRECONDITIONS.{key}",
                )

    def _validate_sinks_and_monitor(
        self, G: TNFRGraph, kw: Mapping[str, Any]
    ) -> None:
        """Reject malformed active telemetry targets before IL dispatch."""

        for key in (
            "IL_coherence_tracking",
            "IL_dnfr_reductions",
            "IL_phase_locking",
            "recognized_coherence_patterns",
        ):
            require_list_sink(G.graph, key, operator=self.name)

        collect_metrics = bool(kw.get("collect_metrics", False)) or bool(
            G.graph.get("COLLECT_OPERATOR_METRICS", False)
        )
        if collect_metrics:
            require_list_sink(G.graph, "operator_metrics", operator=self.name)

        monitor = G.graph.get("integrity_monitor")
        if monitor is not None and not all(
            callable(getattr(monitor, method, None))
            for method in ("before_operator", "after_operator")
        ):
            reject_operator_argument(
                self.name,
                "integrity_monitor must provide callable before_operator and "
                "after_operator methods",
            )

    def _build_proposal(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> CoherenceStageProposal:
        """Materialize the complete IL proposal before glyph dispatch."""

        factors = resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        return propose_coherence_stage(
            G,
            node,
            factors["IL_dnfr_factor"],
            radius=kw.get("coherence_radius", 1),
            phase_locking_coefficient=kw.get(
                "phase_locking_coefficient", 0.3
            ),
        )

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Commit one kernel proposal through the canonical direct lifecycle."""

        proposal = self._build_proposal(G, node, **kw)
        super()._execute(
            G,
            node,
            _coherence_stage_proposal=proposal,
            **kw,
        )

    def _after_glyph_application(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        """Commit phase and IL telemetry before monitor and metric hooks."""

        proposal = kw.get("_coherence_stage_proposal")
        if not isinstance(proposal, CoherenceStageProposal):
            raise RuntimeError("Coherence execution lacks its bound proposal")
        if proposal.node != node:
            raise RuntimeError("Coherence proposal target changed")

        dnfr_after = finite_node_real(
            G.nodes[node],
            ALIAS_DNFR,
            0.0,
            operator=self.name,
            label="DeltaNFR result",
        )
        if dnfr_after != proposal.dnfr_after:
            raise RuntimeError("Coherence pressure commit diverged from its proposal")

        self._commit_phase_proposal(G, node, proposal.phase)
        G.graph.setdefault("IL_coherence_tracking", []).append(
            coherence_tracking_event(G, proposal)
        )
        G.graph.setdefault("IL_dnfr_reductions", []).append(
            coherence_reduction_event(proposal, dnfr_after)
        )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate IL-specific preconditions."""
        from .preconditions import validate_coherence

        validate_coherence(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect IL-specific metrics."""
        from .metrics import coherence_metrics

        return coherence_metrics(G, node, state_before["dnfr"])

    def _apply_phase_locking(
        self, G: TNFRGraph, node: Any, locking_coefficient: float = 0.3
    ) -> None:
        """Align one node phase through the shared pure circular kernel."""

        coefficient = finite_real(
            locking_coefficient,
            operator=self.name,
            label="phase_locking_coefficient",
            lower=0.0,
            upper=1.0,
        )
        require_list_sink(G.graph, "IL_phase_locking", operator=self.name)
        proposal = propose_coherence_phase(G, node, coefficient)
        self._commit_phase_proposal(G, node, proposal)

    def _commit_phase_proposal(
        self, G: TNFRGraph, node: Any, proposal: CoherencePhaseProposal
    ) -> None:
        """Commit an already checked IL phase result and ordered telemetry."""

        set_attr(G.nodes[node], ALIAS_THETA, proposal.theta_after)
        if not proposal.has_neighbors:
            return
        G.graph.setdefault("IL_phase_locking", []).append(
            {
                "node": node,
                "theta_before": proposal.theta_before,
                "theta_after": proposal.theta_after,
                "theta_network": proposal.theta_network,
                "delta_theta": proposal.delta_theta,
                "alignment_achieved": abs(proposal.delta_theta)
                * (1.0 - proposal.coefficient),
            }
        )


# The immutable stage bypasses the direct lifecycle only for this exact
# implementation. Keep a stable identity so monkeypatches and subclasses fall
# back to the public operator path inside the outer stage transaction.
_CANONICAL_COHERENCE_EXECUTE = Coherence._execute
