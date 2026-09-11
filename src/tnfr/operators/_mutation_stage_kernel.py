"""Pure immutable proposals for the ZHIR runtime and network stage.

The low-level proposal owns Mutation's phase map and glyph telemetry.  The
network proposal additionally freezes the accepted temporal evidence,
structural acceleration, U4b destabilizer context and bifurcation decision
from one stage-start graph.  Neither proposal mutates graph state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import glyph_history
from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_EPI_KIND, ALIAS_THETA
from ..constants.canonical import INV_PI
from ..errors import TNFRValueError
from ..types import Glyph
from ._argument_validation import (
    finite_real,
    reject_operator_argument,
    require_list_sink,
)
from ._mutation_gate import MutationRuntimeGate, validate_mutation_runtime_gate

__all__ = [
    "MutationNetworkStageProposal",
    "MutationStageProposal",
    "commit_mutation_bifurcation_event",
    "commit_mutation_lifecycle",
    "commit_mutation_structure",
    "emit_mutation_lifecycle_log",
    "propose_mutation_network_stage",
    "propose_mutation_stage",
]


_TelemetryValue = bool | float | int


def _finite_scalar(value: Any, label: str) -> float:
    """Match the low-level glyph boundary for finite binary64 values."""

    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} must be representable as a finite scalar",
            context={"field": label, "value": repr(value)},
        ) from exc
    if not math.isfinite(resolved):
        raise TNFRValueError(
            f"{label} must remain finite",
            context={"field": label, "value": repr(value)},
        )
    return resolved


@dataclass(frozen=True, slots=True)
class MutationStageProposal:
    """One validated ZHIR phase result and its immutable telemetry payload."""

    theta_before: float
    theta_after: float
    theta_shift: float
    fixed_mode: bool
    regime_changed: bool | None
    regime_before: int | None
    regime_after: int | None

    @property
    def telemetry_items(self) -> tuple[tuple[str, _TelemetryValue], ...]:
        """Return exactly the branch-specific metadata written by ZHIR."""

        common: tuple[tuple[str, _TelemetryValue], ...] = (
            ("_zhir_theta_shift", self.theta_shift),
        )
        if self.fixed_mode:
            return common + (("_zhir_fixed_mode", True),)
        assert self.regime_changed is not None
        assert self.regime_before is not None
        assert self.regime_after is not None
        return common + (
            ("_zhir_theta_before", self.theta_before),
            ("_zhir_theta_after", self.theta_after),
            ("_zhir_regime_changed", self.regime_changed),
            ("_zhir_regime_before", self.regime_before),
            ("_zhir_regime_after", self.regime_after),
            ("_zhir_fixed_mode", False),
        )


@dataclass(frozen=True, slots=True)
class MutationNetworkStageProposal:
    """One target-bound ZHIR decision frozen at the stage-start state."""

    node: Any
    phase: MutationStageProposal
    runtime_gate: MutationRuntimeGate
    structural_acceleration: float
    acceleration_magnitude: float
    tau: float
    bifurcation_potential: bool
    destabilizer_operator: str | None
    destabilizer_distance: int | None
    recent_history: tuple[str, ...]
    epi_kind_before: str | None
    operator_step: int
    glyph: Glyph = field(default=Glyph.ZHIR, init=False)

    @property
    def mutation_context(self) -> dict[str, Any]:
        """Materialize the accepted U4b context for runtime telemetry."""

        return {
            "destabilizer_operator": self.destabilizer_operator,
            "destabilizer_distance": self.destabilizer_distance,
            "recent_history": list(self.recent_history),
        }


def propose_mutation_stage(
    theta: Any,
    delta_nfr: Any = None,
    *,
    theta_shift_factor: Any = INV_PI,
    fixed_shift: Any = None,
) -> MutationStageProposal:
    """Return the RNG-free ZHIR phase proposal without mutating runtime state.

    ``fixed_shift is not None`` selects the backward-compatible fixed branch.
    That branch deliberately does not read ``delta_nfr`` or
    ``theta_shift_factor``, matching the active-factor runtime contract.
    """

    theta_before = _finite_scalar(theta, "ZHIR phase state") % math.tau
    if fixed_shift is not None:
        shift = _finite_scalar(fixed_shift, "ZHIR phase shift")
        theta_after = _finite_scalar(
            (theta_before + (shift % math.tau)) % math.tau,
            "ZHIR phase proposal",
        )
        return MutationStageProposal(
            theta_before=theta_before,
            theta_after=theta_after,
            theta_shift=shift,
            fixed_mode=True,
            regime_changed=None,
            regime_before=None,
            regime_after=None,
        )

    dnfr = _finite_scalar(delta_nfr, "ZHIR DeltaNFR state")
    factor = _finite_scalar(theta_shift_factor, "ZHIR theta shift factor")
    shift = _finite_scalar(
        factor * math.copysign(1.0, dnfr) * (math.pi / 4.0),
        "ZHIR phase shift",
    )
    theta_after = _finite_scalar(
        (theta_before + (shift % math.tau)) % math.tau,
        "ZHIR phase proposal",
    )
    regime_before = int(theta_before // (math.pi / 2.0))
    regime_after = int(theta_after // (math.pi / 2.0))
    return MutationStageProposal(
        theta_before=theta_before,
        theta_after=theta_after,
        theta_shift=shift,
        fixed_mode=False,
        regime_changed=regime_before != regime_after,
        regime_before=regime_before,
        regime_after=regime_after,
    )


def propose_mutation_network_stage(
    graph: Any,
    node: Any,
    factors: Mapping[str, Any],
    *,
    tau: Any = None,
) -> MutationNetworkStageProposal:
    """Freeze one complete ZHIR decision without changing ``graph``.

    The caller is responsible for grammar selection.  This function binds the
    non-disableable growth gate, the active phase-factor branch, acceleration
    telemetry and the pre-ZHIR destabilizer context to one graph snapshot.
    """

    gate = validate_mutation_runtime_gate(graph.nodes[node], graph.graph)
    if "ZHIR_theta_shift" in factors:
        phase = propose_mutation_stage(
            get_attr(graph.nodes[node], ALIAS_THETA, 0.0, strict=True),
            fixed_shift=factors["ZHIR_theta_shift"],
        )
    else:
        phase = propose_mutation_stage(
            get_attr(graph.nodes[node], ALIAS_THETA, 0.0, strict=True),
            get_attr(graph.nodes[node], ALIAS_DNFR, 0.0, strict=True),
            theta_shift_factor=factors["ZHIR_theta_shift_factor"],
        )

    from .nodal_equation import compute_d2epi_dt2
    from .preconditions.mutation import record_destabilizer_context

    structural_acceleration = float(
        compute_d2epi_dt2(graph, node, store=False)
    )
    acceleration_magnitude = abs(structural_acceleration)
    tau_raw = tau
    if tau_raw is None:
        tau_raw = graph.graph.get(
            "BIFURCATION_THRESHOLD_TAU",
            graph.graph.get("ZHIR_BIFURCATION_THRESHOLD", 0.5),
        )
    resolved_tau = finite_real(
        tau_raw,
        operator="Mutation",
        label="tau",
        lower=0.0,
    )
    bifurcation_potential = acceleration_magnitude > resolved_tau
    mode = graph.graph.get("ZHIR_BIFURCATION_MODE", "detection")
    if mode != "detection":
        reject_operator_argument(
            "Mutation",
            "ZHIR_BIFURCATION_MODE supports 'detection' only; use THOL "
            "for variant or sub-EPI creation",
        )
    if bifurcation_potential:
        require_list_sink(
            graph.graph,
            "zhir_bifurcation_events",
            operator="Mutation",
        )

    context = record_destabilizer_context(
        graph,
        node,
        record=False,
        emit_log=False,
    )
    raw_kind = get_attr(
        graph.nodes[node],
        ALIAS_EPI_KIND,
        None,
        strict=True,
        conv=lambda value: None if value is None else str(value),
    )
    return MutationNetworkStageProposal(
        node=node,
        phase=phase,
        runtime_gate=gate,
        structural_acceleration=structural_acceleration,
        acceleration_magnitude=acceleration_magnitude,
        tau=resolved_tau,
        bifurcation_potential=bifurcation_potential,
        destabilizer_operator=context.get("destabilizer_operator"),
        destabilizer_distance=context.get("destabilizer_distance"),
        recent_history=tuple(context.get("recent_history", ())),
        epi_kind_before=raw_kind,
        operator_step=glyph_history.next_operator_step(graph.nodes[node]),
    )


def commit_mutation_structure(
    graph: Any, proposal: MutationNetworkStageProposal
) -> None:
    """Commit ZHIR's disjoint phase channel and glyph telemetry."""

    from ..node import NodeNX

    data = graph.nodes[proposal.node]
    NodeNX.from_graph(graph, proposal.node).theta = proposal.phase.theta_after
    for key, value in proposal.phase.telemetry_items:
        data[key] = value


def commit_mutation_lifecycle(
    graph: Any, proposal: MutationNetworkStageProposal
) -> None:
    """Commit proposal-bound diagnostics after ZHIR history is recorded."""

    data = graph.nodes[proposal.node]
    observed_step = glyph_history.current_operator_step(data)
    if observed_step != proposal.operator_step:
        raise RuntimeError(
            "Mutation proposal operator step changed before lifecycle commit"
        )
    data["_mutation_context"] = proposal.mutation_context
    set_attr(data, ALIAS_D2EPI, proposal.structural_acceleration)
    data["_zhir_bifurcation_potential"] = proposal.bifurcation_potential
    data["_zhir_d2epi"] = proposal.acceleration_magnitude
    data["_zhir_tau"] = proposal.tau
    data["_zhir_operator_step"] = proposal.operator_step
    data["_zhir_gate_depi_dt"] = proposal.runtime_gate.threshold.depi_dt
    data["_zhir_gate_xi"] = proposal.runtime_gate.threshold.xi
    data["_zhir_gate_history_key"] = proposal.runtime_gate.threshold.history_key
    data["_zhir_gate_sample_interval"] = (
        proposal.runtime_gate.threshold.sample_interval
    )
    commit_mutation_bifurcation_event(graph, proposal)


def emit_mutation_lifecycle_log(
    proposal: MutationNetworkStageProposal,
) -> None:
    """Publish U4b context only after the surrounding transaction succeeds."""

    from .preconditions.mutation import emit_destabilizer_context_log

    emit_destabilizer_context_log(proposal.node, proposal.mutation_context)


def commit_mutation_bifurcation_event(
    graph: Any, proposal: MutationNetworkStageProposal
) -> None:
    """Append the accepted ZHIR event after operator history is committed."""

    if not proposal.bifurcation_potential:
        return
    events = graph.graph.setdefault("zhir_bifurcation_events", [])
    events.append(
        {
            "node": proposal.node,
            "d2_epi": proposal.acceleration_magnitude,
            "tau": proposal.tau,
            "timestamp": proposal.operator_step,
            "event_index": len(events) + 1,
        }
    )
