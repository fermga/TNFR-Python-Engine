"""Coherence (IL) operator.

Purpose: stabilize form; reduce delta NFR; raise coherence.
Physics: negative feedback drives delta NFR toward 0.
Grammar: stabilizer (U2); safe closure component.
Effects: pressure lowers; EPI preserved; optional phase locking.
Preconditions: active structure; prior destabilizer if recent IL already.
Typical: AL->UM->IL; OZ->IL; VAL->IL. Avoid redundant IL chains.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..alias import get_attr, set_attr
from ..config.operator_names import COHERENCE
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..metrics.trig import neighbor_phase_mean_list
from ..types import Glyph, TNFRGraph
from ..utils import angle_diff
from ._argument_validation import (
    finite_node_real,
    finite_real,
    nonnegative_integer,
    reject_operator_argument,
    require_list_sink,
    strict_bool,
)
from .definitions_base import Operator
from .factor_contracts import resolve_runtime_operator_factors


@dataclass(frozen=True, slots=True)
class _PhaseLockProposal:
    """A fully checked IL phase update, prepared before glyph dispatch."""

    theta_before: float
    theta_after: float
    theta_network: float | None
    delta_theta: float
    coefficient: float
    has_neighbors: bool


@dataclass(frozen=True, slots=True)
class _CoherenceProposal:
    """All public IL values needed before the low-level state change."""

    radius: int
    dnfr_before: float
    dnfr_after: float
    coherence_global_before: float
    coherence_local_before: float
    phase: _PhaseLockProposal


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
        super()._validate_application_preconditions(G, node, **kw)

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
            EPI_IL_MAX,
            EPI_IL_MIN,
            VF_IL_MIN,
        )

        raw = G.graph.get("IL_PRECONDITIONS", {})
        if not isinstance(raw, Mapping):
            reject_operator_argument(
                self.name, "IL_PRECONDITIONS must be a mapping"
            )
        min_epi = finite_real(
            raw.get("min_epi", EPI_IL_MIN),
            operator=self.name,
            label="IL_PRECONDITIONS.min_epi",
        )
        max_epi = finite_real(
            raw.get("max_epi", EPI_IL_MAX),
            operator=self.name,
            label="IL_PRECONDITIONS.max_epi",
        )
        if max_epi <= min_epi:
            reject_operator_argument(
                self.name,
                "IL_PRECONDITIONS.max_epi must be greater than min_epi",
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

    def _build_phase_proposal(
        self, G: TNFRGraph, node: Any, coefficient: float
    ) -> _PhaseLockProposal:
        """Prepare a finite normalized phase result without creating caches."""

        theta_before = finite_node_real(
            G.nodes[node],
            ALIAS_THETA,
            0.0,
            operator=self.name,
            label="theta state",
        )
        theta_normalized = finite_real(
            theta_before % math.tau,
            operator=self.name,
            label="normalized theta state",
            lower=0.0,
            upper=math.tau,
        )
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return _PhaseLockProposal(
                theta_before=theta_before,
                theta_after=theta_normalized,
                theta_network=None,
                delta_theta=0.0,
                coefficient=coefficient,
                has_neighbors=False,
            )

        phases = {
            neighbor: finite_node_real(
                G.nodes[neighbor],
                ALIAS_THETA,
                0.0,
                operator=self.name,
                label=f"neighbor theta state for {neighbor!r}",
            )
            for neighbor in neighbors
        }
        cosines = {neighbor: math.cos(phase) for neighbor, phase in phases.items()}
        sines = {neighbor: math.sin(phase) for neighbor, phase in phases.items()}
        theta_network = finite_real(
            neighbor_phase_mean_list(
                neighbors, cosines, sines, fallback=theta_normalized
            )
            % math.tau,
            operator=self.name,
            label="neighborhood phase mean",
            lower=0.0,
            upper=math.tau,
        )
        delta_theta = finite_real(
            angle_diff(theta_network, theta_normalized),
            operator=self.name,
            label="phase-locking delta",
        )
        theta_after = finite_real(
            (theta_normalized + coefficient * delta_theta) % math.tau,
            operator=self.name,
            label="phase-locking proposal",
            lower=0.0,
            upper=math.tau,
        )
        return _PhaseLockProposal(
            theta_before=theta_before,
            theta_after=theta_after,
            theta_network=theta_network,
            delta_theta=delta_theta,
            coefficient=coefficient,
            has_neighbors=True,
        )

    def _build_proposal(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> _CoherenceProposal:
        """Materialize the complete IL proposal before the glyph handler."""

        from ..metrics.coherence import (
            compute_global_coherence,
            compute_local_coherence,
        )

        radius = nonnegative_integer(
            kw.get("coherence_radius", 1),
            operator=self.name,
            label="coherence_radius",
        )
        coefficient = finite_real(
            kw.get("phase_locking_coefficient", 0.3),
            operator=self.name,
            label="phase_locking_coefficient",
            lower=0.0,
            upper=1.0,
        )
        for candidate in G.nodes:
            finite_node_real(
                G.nodes[candidate],
                ALIAS_DNFR,
                0.0,
                operator=self.name,
                label=f"DeltaNFR state for {candidate!r}",
            )

        dnfr_before = finite_node_real(
            G.nodes[node],
            ALIAS_DNFR,
            0.0,
            operator=self.name,
            label="DeltaNFR state",
        )
        factors = resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        dnfr_after = finite_real(
            dnfr_before * factors["IL_dnfr_factor"],
            operator=self.name,
            label="DeltaNFR proposal",
        )
        coherence_global_before = finite_real(
            compute_global_coherence(G),
            operator=self.name,
            label="global coherence telemetry",
            lower=0.0,
            upper=1.0,
        )
        coherence_local_before = finite_real(
            compute_local_coherence(G, node, radius=radius),
            operator=self.name,
            label="local coherence telemetry",
            lower=0.0,
            upper=1.0,
        )
        return _CoherenceProposal(
            radius=radius,
            dnfr_before=dnfr_before,
            dnfr_after=dnfr_after,
            coherence_global_before=coherence_global_before,
            coherence_local_before=coherence_local_before,
            phase=self._build_phase_proposal(G, node, coefficient),
        )

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Reduce delta NFR; track coherence; optional phase lock.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Optional keys:
            - coherence_radius: local coherence radius (default 1)
            - phase_locking_coefficient: phase lock strength in [0,1]

        Notes
        -----
        Canonical effect: dnfr -> dnfr * (1 - rho) rho≈0.3.

        Reduction applied by grammar layer (dnfr_factor). Adds telemetry.

        **C(t) Coherence Tracking:**

        Captures global/local coherence before & after:
        - C_global: 1 - (sigma_dnfr / max|dnfr|)
        - C_local: neighborhood coherence (radius)

        Stored in G.graph['IL_coherence_tracking'].

        Phase Locking:

        Phase align:
        - theta_new = theta + a*(theta_net - theta)
        Stored in telemetry list.

        Adjust rho via GLYPH_FACTORS['IL_dnfr_factor'] (default 0.7).
        """
        from ..metrics.coherence import (
            compute_global_coherence,
            compute_local_coherence,
        )

        proposal = self._build_proposal(G, node, **kw)
        super()._execute(G, node, **kw)
        self._commit_phase_proposal(G, node, proposal.phase)

        coherence_global_after = finite_real(
            compute_global_coherence(G),
            operator=self.name,
            label="global coherence result",
            lower=0.0,
            upper=1.0,
        )
        coherence_local_after = finite_real(
            compute_local_coherence(G, node, radius=proposal.radius),
            operator=self.name,
            label="local coherence result",
            lower=0.0,
            upper=1.0,
        )
        dnfr_after = finite_node_real(
            G.nodes[node],
            ALIAS_DNFR,
            0.0,
            operator=self.name,
            label="DeltaNFR result",
        )

        G.graph.setdefault("IL_coherence_tracking", []).append(
            {
                "node": node,
                "C_global_before": proposal.coherence_global_before,
                "C_global_after": coherence_global_after,
                "C_global_delta": coherence_global_after
                - proposal.coherence_global_before,
                "C_local_before": proposal.coherence_local_before,
                "C_local_after": coherence_local_after,
                "C_local_delta": coherence_local_after
                - proposal.coherence_local_before,
            }
        )

        if proposal.dnfr_before > 0:
            actual_reduction_factor = (
                proposal.dnfr_before - dnfr_after
            ) / proposal.dnfr_before
        else:
            actual_reduction_factor = 0.0

        G.graph.setdefault("IL_dnfr_reductions", []).append(
            {
                "node": node,
                "before": proposal.dnfr_before,
                "after": dnfr_after,
                "reduction": proposal.dnfr_before - dnfr_after,
                "reduction_factor": actual_reduction_factor,
            }
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
        """Align node phase with neighborhood mean.

        Parameters
        ----------
        locking_coefficient : float
            Phase alignment strength a (default 0.3).

        Notes
        -----
        **Canonical Specification:**

        Steps:
        1. Network phase = circular mean of neighbors
        2. delta = shortest angular diff
        3. theta_new = theta + a*delta
        4. normalize to [0,2*pi]

        **Circular Statistics:**

        Circular mean via complex exponential averaging.

        Handles wrap-around (e.g., 0.1 & 6.2 -> ~0).

        **Telemetry:**

        Telemetry stored in G.graph['IL_phase_locking'].

        **Special Cases:**

        - No neighbors: Circular phase unchanged; representative normalized
        - Single neighbor: Aligns toward that neighbor's phase
        - Isolated node: No locking telemetry is added

        See Also
        --------
        metrics.phase_coherence.compute_phase_alignment: phase alignment
        """
        coefficient = finite_real(
            locking_coefficient,
            operator=self.name,
            label="phase_locking_coefficient",
            lower=0.0,
            upper=1.0,
        )
        require_list_sink(G.graph, "IL_phase_locking", operator=self.name)
        proposal = self._build_phase_proposal(G, node, coefficient)
        self._commit_phase_proposal(G, node, proposal)

    def _commit_phase_proposal(
        self, G: TNFRGraph, node: Any, proposal: _PhaseLockProposal
    ) -> None:
        """Commit an already checked IL phase result and its telemetry."""

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
