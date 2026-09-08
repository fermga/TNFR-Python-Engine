"""Immutable all-target proposal and reduction kernel for Dissonance (OZ).

The local OZ action and every outgoing propagation read one stage-start graph.
For each node, the simultaneous pressure is its local proposal when targeted
(or its snapshot pressure otherwise) plus all incoming propagated increments.
Incoming values are reduced in immutable snapshot-node rank with ``math.fsum``.
This is an explicit engine policy, not a general superposition theorem for
arbitrary structural-pressure laws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..dynamics.propagation import _prepare_dissonance_propagation
from ..errors import TNFRValueError
from ..types import Glyph
from .dissonance import _plan_local_dissonance


@dataclass(frozen=True, slots=True)
class DissonanceTargetProposal:
    """One snapshot-bound local OZ proposal and RNG progress transition."""

    node: Any
    glyph: Glyph
    dnfr_before: float
    dnfr_local_after: float
    local_magnitude: float
    had_jitter_progress: bool
    jitter_progress_before: Any
    has_jitter_progress_after: bool
    jitter_progress_after: Any
    precondition_context: Mapping[str, Any] | None

    @property
    def writes_local_pressure(self) -> bool:
        """Whether the low-level local branch materialized a pressure write."""

        return bool(
            self.local_magnitude != 0.0
            or self.had_jitter_progress != self.has_jitter_progress_after
            or self.jitter_progress_before != self.jitter_progress_after
        )

    @property
    def local_contract_satisfied(self) -> bool:
        """Whether local OZ did not reduce pressure magnitude."""

        return abs(self.dnfr_local_after) >= abs(self.dnfr_before) - 1e-9


@dataclass(frozen=True, slots=True)
class DissonanceContribution:
    """One positive outgoing pressure increment from a target source."""

    source: Any
    neighbor: Any
    magnitude: float
    phase_weight: float
    coupling_weight: float

    def event(self) -> dict[str, Any]:
        """Return the established neighbor-telemetry record."""

        return {
            "from_node": self.source,
            "magnitude": self.magnitude,
            "phase_weight": self.phase_weight,
            "coupling_weight": self.coupling_weight,
        }


@dataclass(frozen=True, slots=True)
class DissonancePressureUpdate:
    """One deterministically reduced final nodal pressure."""

    node: Any
    dnfr_before: float
    dnfr_after: float
    incoming_total: float
    targeted: bool
    write_pressure: bool


@dataclass(frozen=True, slots=True)
class DissonanceGraphEvent:
    """One source-level propagation summary in requested target order."""

    source: Any
    magnitude: float
    affected_nodes: tuple[Any, ...]

    def as_record(self) -> dict[str, Any]:
        """Return the established graph-level telemetry record."""

        return {
            "source": self.source,
            "magnitude": self.magnitude,
            "affected_nodes": list(self.affected_nodes),
            "affected_count": len(self.affected_nodes),
        }


@dataclass(frozen=True, slots=True)
class DissonanceStageProposal:
    """Complete immutable OZ stage proposal before any live write."""

    targets: tuple[Any, ...]
    target_proposals: tuple[DissonanceTargetProposal, ...]
    pressure_updates: tuple[DissonancePressureUpdate, ...]
    contributions: tuple[DissonanceContribution, ...]
    graph_events: tuple[DissonanceGraphEvent, ...]


def _snapshot_pressure(snapshot: Any, node: Any) -> float:
    """Read one finite signed pressure without permissive fallback."""

    try:
        raw = get_attr(
            snapshot.nodes[node],
            ALIAS_DNFR,
            0.0,
            strict=True,
            conv=lambda value: value,
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"OZ snapshot pressure {node!r} must be a finite real scalar"
        ) from exc
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
        raise TNFRValueError(
            f"OZ snapshot pressure {node!r} must be a finite real scalar"
        )
    try:
        value = float(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"OZ snapshot pressure {node!r} must be a finite real scalar"
        ) from exc
    if not math.isfinite(value):
        raise TNFRValueError(f"OZ snapshot pressure {node!r} must be finite")
    return value


def propose_dissonance_stage(
    snapshot: Any,
    targets: Sequence[Any],
    *,
    propagate: bool,
    propagation_mode: str = "phase_weighted",
    preconditions_validated: bool = False,
) -> DissonanceStageProposal:
    """Build the complete simultaneous OZ pressure and telemetry proposal.

    Local target proposals retain requested target order for lifecycle streams.
    Incoming pressure is instead summed by immutable snapshot-node rank, so the
    structural result cannot depend on the requested target order. The local
    OZ magnitude postcondition is reported separately from the merged pressure:
    signed incoming increments may cancel a negative local proposal.
    """

    targets_tuple = tuple(targets)
    rank = {node: index for index, node in enumerate(snapshot.nodes)}
    if len(rank) != len(snapshot):
        raise RuntimeError("OZ snapshot node rank is not injective")

    local_proposals: list[DissonanceTargetProposal] = []
    local_by_node: dict[Any, DissonanceTargetProposal] = {}
    for node in targets_tuple:
        snapshot_pressure = _snapshot_pressure(snapshot, node)
        plan = _plan_local_dissonance(snapshot, node)
        if plan.dnfr_before != snapshot_pressure:
            raise RuntimeError(
                f"OZ local plan changed snapshot pressure for target {node!r}"
            )
        proposal = DissonanceTargetProposal(
            node=node,
            glyph=Glyph.OZ,
            dnfr_before=plan.dnfr_before,
            dnfr_local_after=plan.dnfr_after,
            local_magnitude=plan.magnitude,
            had_jitter_progress=plan.had_jitter_progress,
            jitter_progress_before=plan.jitter_progress_before,
            has_jitter_progress_after=plan.has_jitter_progress_after,
            jitter_progress_after=plan.jitter_progress_after,
            precondition_context=(
                deepcopy(
                    snapshot.nodes[node]["_oz_precondition_context"]
                )
                if preconditions_validated
                else None
            ),
        )
        local_proposals.append(proposal)
        local_by_node[node] = proposal

    contributions: list[DissonanceContribution] = []
    graph_events: list[DissonanceGraphEvent] = []
    for proposal in local_proposals:
        if not propagate or proposal.local_magnitude == 0.0:
            continue
        propagation = _prepare_dissonance_propagation(
            snapshot,
            proposal.node,
            proposal.local_magnitude,
            propagation_mode=propagation_mode,
            dnfr_overrides={proposal.node: proposal.dnfr_local_after},
        )
        affected_nodes = tuple(sorted(propagation.affected, key=rank.__getitem__))
        graph_events.append(
            DissonanceGraphEvent(
                source=proposal.node,
                magnitude=proposal.local_magnitude,
                affected_nodes=affected_nodes,
            )
        )
        for propagated in propagation.proposals:
            event = propagated.event
            contributions.append(
                DissonanceContribution(
                    source=proposal.node,
                    neighbor=propagated.neighbor,
                    magnitude=float(event["magnitude"]),
                    phase_weight=float(event["phase_weight"]),
                    coupling_weight=float(event["coupling_weight"]),
                )
            )

    incoming: dict[Any, list[DissonanceContribution]] = {}
    for contribution in contributions:
        incoming.setdefault(contribution.neighbor, []).append(contribution)
    for values in incoming.values():
        values.sort(key=lambda item: rank[item.source])

    touched = set(targets_tuple) | set(incoming)
    pressure_updates: list[DissonancePressureUpdate] = []
    for node in sorted(touched, key=rank.__getitem__):
        local = local_by_node.get(node)
        before = local.dnfr_before if local is not None else _snapshot_pressure(
            snapshot, node
        )
        base = local.dnfr_local_after if local is not None else before
        additions = [item.magnitude for item in incoming.get(node, ())]
        try:
            incoming_total = math.fsum(additions)
            after = math.fsum((base, *additions))
        except OverflowError as exc:
            raise TNFRValueError(
                f"OZ reduced pressure {node!r} must be finite"
            ) from exc
        if not math.isfinite(incoming_total) or not math.isfinite(after):
            raise TNFRValueError(f"OZ reduced pressure {node!r} must be finite")
        pressure_updates.append(
            DissonancePressureUpdate(
                node=node,
                dnfr_before=before,
                dnfr_after=after,
                incoming_total=incoming_total,
                targeted=local is not None,
                write_pressure=bool(additions)
                or bool(local is not None and local.writes_local_pressure),
            )
        )

    return DissonanceStageProposal(
        targets=targets_tuple,
        target_proposals=tuple(local_proposals),
        pressure_updates=tuple(pressure_updates),
        contributions=tuple(contributions),
        graph_events=tuple(graph_events),
    )


__all__ = [
    "DissonanceContribution",
    "DissonanceGraphEvent",
    "DissonancePressureUpdate",
    "DissonanceStageProposal",
    "DissonanceTargetProposal",
    "propose_dissonance_stage",
]
