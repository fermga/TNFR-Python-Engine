"""Pure immutable proposal and telemetry kernel for Coherence (IL).

The kernel reads a graph state and returns frozen values.  It never mutates the
input graph, emits warnings, appends telemetry, or advances lifecycle state.
Direct node execution and simultaneous all-target execution therefore share the
same pressure contraction, circular phase lock, and coherence definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA
from ..metrics.trig import neighbor_phase_mean_list
from ..types import Glyph
from ..utils import angle_diff
from ._argument_validation import finite_node_real, finite_real, nonnegative_integer


@dataclass(frozen=True, slots=True)
class CoherencePhaseProposal:
    """Finite circular phase-lock proposal for one target."""

    theta_before: float
    theta_after: float
    theta_network: float | None
    delta_theta: float
    coefficient: float
    has_neighbors: bool


@dataclass(frozen=True, slots=True)
class CoherenceGlobalReadout:
    """Canonical and auxiliary global coherence read-outs."""

    structural: float
    dispersion: float


@dataclass(frozen=True, slots=True)
class CoherenceReadout:
    """Global and radius-local coherence read-outs at one instant."""

    structural_global: float
    structural_local: float
    dispersion_global: float
    dispersion_local: float


@dataclass(frozen=True, slots=True)
class CoherenceStageProposal:
    """Complete target-bound IL proposal from one immutable graph state."""

    node: Any
    glyph: Glyph
    radius: int
    dnfr_before: float
    dnfr_after: float
    phase: CoherencePhaseProposal
    coherence_before: CoherenceReadout
    precondition_warnings: tuple[str, ...] = ()

def capture_coherence_globals(graph: Any) -> CoherenceGlobalReadout:
    """Read canonical structural C(t) and auxiliary pressure dispersion."""

    from ..metrics.coherence import compute_global_coherence
    from ..metrics.common import compute_coherence

    return CoherenceGlobalReadout(
        structural=finite_real(
            compute_coherence(graph),
            operator="Coherence",
            label="global structural coherence",
            lower=0.0,
            upper=1.0,
        ),
        dispersion=finite_real(
            compute_global_coherence(graph),
            operator="Coherence",
            label="global pressure-dispersion coherence",
            lower=0.0,
            upper=1.0,
        ),
    )


def capture_coherence_readout(
    graph: Any,
    node: Any,
    radius: int,
    *,
    global_readout: CoherenceGlobalReadout | None = None,
) -> CoherenceReadout:
    """Read global and graph-ball coherence without changing graph caches."""

    from ..metrics.coherence import compute_local_coherence
    from ..metrics.local_coherence import compute_radius_structural_coherence

    radius = nonnegative_integer(
        radius,
        operator="Coherence",
        label="coherence_radius",
    )
    globals_now = global_readout or capture_coherence_globals(graph)
    structural_local = finite_real(
        compute_radius_structural_coherence(graph, node, radius),
        operator="Coherence",
        label="radius-local structural coherence",
        lower=0.0,
        upper=1.0,
    )
    dispersion_local = finite_real(
        compute_local_coherence(graph, node, radius=radius),
        operator="Coherence",
        label="radius-local pressure-dispersion coherence",
        lower=0.0,
        upper=1.0,
    )
    return CoherenceReadout(
        structural_global=globals_now.structural,
        structural_local=structural_local,
        dispersion_global=globals_now.dispersion,
        dispersion_local=dispersion_local,
    )


def propose_coherence_phase(
    graph: Any,
    node: Any,
    coefficient: float,
) -> CoherencePhaseProposal:
    """Return the snapshot-bound circular phase proposal."""

    coefficient = finite_real(
        coefficient,
        operator="Coherence",
        label="phase_locking_coefficient",
        lower=0.0,
        upper=1.0,
    )
    theta_before = finite_node_real(
        graph.nodes[node],
        ALIAS_THETA,
        0.0,
        operator="Coherence",
        label="theta state",
    )
    theta_normalized = finite_real(
        theta_before % math.tau,
        operator="Coherence",
        label="normalized theta state",
        lower=0.0,
        upper=math.tau,
    )
    neighbors = tuple(graph.neighbors(node))
    if not neighbors:
        return CoherencePhaseProposal(
            theta_before=theta_before,
            theta_after=theta_normalized,
            theta_network=None,
            delta_theta=0.0,
            coefficient=coefficient,
            has_neighbors=False,
        )

    phases = {
        neighbor: finite_node_real(
            graph.nodes[neighbor],
            ALIAS_THETA,
            0.0,
            operator="Coherence",
            label=f"neighbor theta state for {neighbor!r}",
        )
        for neighbor in neighbors
    }
    cosines = {neighbor: math.cos(phase) for neighbor, phase in phases.items()}
    sines = {neighbor: math.sin(phase) for neighbor, phase in phases.items()}
    theta_network = finite_real(
        neighbor_phase_mean_list(
            neighbors,
            cosines,
            sines,
            fallback=theta_normalized,
        )
        % math.tau,
        operator="Coherence",
        label="neighborhood phase mean",
        lower=0.0,
        upper=math.tau,
    )
    delta_theta = finite_real(
        angle_diff(theta_network, theta_normalized),
        operator="Coherence",
        label="phase-locking delta",
    )
    theta_after = finite_real(
        (theta_normalized + coefficient * delta_theta) % math.tau,
        operator="Coherence",
        label="phase-locking proposal",
        lower=0.0,
        upper=math.tau,
    )
    return CoherencePhaseProposal(
        theta_before=theta_before,
        theta_after=theta_after,
        theta_network=theta_network,
        delta_theta=delta_theta,
        coefficient=coefficient,
        has_neighbors=True,
    )


def propose_coherence_pressure(dnfr: Any, factor: Any) -> tuple[float, float]:
    """Return the finite sign-preserving IL pressure contraction."""

    before = finite_real(dnfr, operator="Coherence", label="DeltaNFR state")
    from .factor_contracts import validate_glyph_factor

    retention = validate_glyph_factor("IL_dnfr_factor", factor)
    after = finite_real(
        retention * before,
        operator="Coherence",
        label="DeltaNFR proposal",
    )
    return before, after


def propose_coherence_stage(
    graph: Any,
    node: Any,
    factor: Any,
    *,
    radius: Any = 1,
    phase_locking_coefficient: Any = 0.3,
    global_before: CoherenceGlobalReadout | None = None,
    precondition_warnings: tuple[str, ...] = (),
) -> CoherenceStageProposal:
    """Build a complete immutable IL proposal without side effects."""

    resolved_radius = nonnegative_integer(
        radius,
        operator="Coherence",
        label="coherence_radius",
    )
    dnfr_before, dnfr_after = propose_coherence_pressure(
        get_attr(graph.nodes[node], ALIAS_DNFR, 0.0, strict=True),
        factor,
    )
    return CoherenceStageProposal(
        node=node,
        glyph=Glyph.IL,
        radius=resolved_radius,
        dnfr_before=dnfr_before,
        dnfr_after=dnfr_after,
        phase=propose_coherence_phase(graph, node, phase_locking_coefficient),
        coherence_before=capture_coherence_readout(
            graph,
            node,
            resolved_radius,
            global_readout=global_before,
        ),
        precondition_warnings=tuple(precondition_warnings),
    )


def coherence_phase_event(proposal: CoherenceStageProposal) -> dict[str, Any] | None:
    """Return ordered phase telemetry, or ``None`` for an isolate."""

    phase = proposal.phase
    if not phase.has_neighbors:
        return None
    return {
        "node": proposal.node,
        "theta_before": phase.theta_before,
        "theta_after": phase.theta_after,
        "theta_network": phase.theta_network,
        "delta_theta": phase.delta_theta,
        "alignment_achieved": abs(phase.delta_theta) * (1.0 - phase.coefficient),
    }


def coherence_reduction_event(
    proposal: CoherenceStageProposal,
    dnfr_after: Any | None = None,
) -> dict[str, Any]:
    """Return sign-correct pressure-magnitude reduction telemetry."""

    after = finite_real(
        proposal.dnfr_after if dnfr_after is None else dnfr_after,
        operator="Coherence",
        label="DeltaNFR result",
    )
    magnitude_before = abs(proposal.dnfr_before)
    magnitude_after = abs(after)
    magnitude_reduction = magnitude_before - magnitude_after
    reduction_factor = (
        magnitude_reduction / magnitude_before if magnitude_before > 0.0 else 0.0
    )
    return {
        "node": proposal.node,
        "before": proposal.dnfr_before,
        "after": after,
        "reduction": magnitude_reduction,
        "reduction_factor": reduction_factor,
        "magnitude_before": magnitude_before,
        "magnitude_after": magnitude_after,
        "magnitude_reduction": magnitude_reduction,
        "signed_delta": after - proposal.dnfr_before,
    }


def coherence_tracking_event(
    graph_after: Any,
    proposal: CoherenceStageProposal,
    *,
    global_after: CoherenceGlobalReadout | None = None,
    scope: str = "direct",
) -> dict[str, Any]:
    """Return canonical C(t) and explicitly auxiliary dispersion telemetry."""

    before = proposal.coherence_before
    after = capture_coherence_readout(
        graph_after,
        proposal.node,
        proposal.radius,
        global_readout=global_after,
    )
    return {
        "node": proposal.node,
        "scope": scope,
        "radius": proposal.radius,
        # Compatibility: these historical keys retain the old pressure-
        # dispersion values. Their explicit aliases and metadata distinguish
        # them from canonical structural C(t).
        "C_global_before": before.dispersion_global,
        "C_global_after": after.dispersion_global,
        "C_global_delta": after.dispersion_global - before.dispersion_global,
        "C_local_before": before.dispersion_local,
        "C_local_after": after.dispersion_local,
        "C_local_delta": after.dispersion_local - before.dispersion_local,
        "C_dispersion_global_before": before.dispersion_global,
        "C_dispersion_global_after": after.dispersion_global,
        "C_dispersion_global_delta": (
            after.dispersion_global - before.dispersion_global
        ),
        "C_dispersion_local_before": before.dispersion_local,
        "C_dispersion_local_after": after.dispersion_local,
        "C_dispersion_local_delta": (
            after.dispersion_local - before.dispersion_local
        ),
        "legacy_C_fields_deprecated": True,
        "legacy_C_fields_definition": "pressure_dispersion_auxiliary",
        # Canonical stage/global and radius-local constitutive coherence.
        "C_t_global_before": before.structural_global,
        "C_t_global_after": after.structural_global,
        "C_t_global_delta": (
            after.structural_global - before.structural_global
        ),
        "C_structural_local_before": before.structural_local,
        "C_structural_local_after": after.structural_local,
        "C_structural_local_delta": (
            after.structural_local - before.structural_local
        ),
    }


__all__ = [
    "CoherenceGlobalReadout",
    "CoherencePhaseProposal",
    "CoherenceReadout",
    "CoherenceStageProposal",
    "capture_coherence_globals",
    "capture_coherence_readout",
    "coherence_phase_event",
    "coherence_reduction_event",
    "coherence_tracking_event",
    "propose_coherence_phase",
    "propose_coherence_pressure",
    "propose_coherence_stage",
]
