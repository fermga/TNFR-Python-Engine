"""Pure snapshot proposals and deterministic merge policy for Coupling (UM).

This module defines an explicit numerical policy of the engine, not a TNFR
theorem. Local proposals read one immutable snapshot. Overlapping phase writes
are merged through snapshot-rank-ordered shortest-arc displacements, and final
phases are normalized to the half-open circular interval.

Functions in this module do not mutate graph, node, history, cache, or telemetry
state. Direct and network execution own their respective commit boundaries.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from ..alias import get_attr
from ..constants import DEFAULTS
from ..constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..constants.canonical import UM_COMPAT_THRESHOLD as _UM_COMPAT_CANONICAL
from ..errors import TNFRValueError
from ..metrics.phase_compatibility import compute_phase_coupling_strength
from ..rng import make_rng
from ..types import Glyph
from ..utils import angle_diff
from ._epi_domain import require_real_scalar_epi
from ._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors

_EPI_SIMILARITY_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class CouplingPhaseProposal:
    """One target's proposed phase write to one node."""

    source: Any
    node: Any
    theta_before: float
    theta_proposed: float


@dataclass(frozen=True, slots=True)
class CouplingLinkCandidate:
    """Snapshot-bound functional-link candidate."""

    source: Any
    target: Any
    epi_source: float
    epi_target: float
    si_source: float
    si_target: float


@dataclass(frozen=True, slots=True)
class CouplingTargetProposal:
    """Immutable local UM proposal from a stage-start snapshot."""

    node: Any
    glyph: Glyph
    compatible_neighbors: tuple[Any, ...]
    effective_phase_limit: float
    phase_proposals: tuple[CouplingPhaseProposal, ...]
    vf_before: float
    vf_after: float
    write_vf: bool
    dnfr_before: float
    dnfr_reduction_factor: float
    write_dnfr: bool
    link_candidates: tuple[CouplingLinkCandidate, ...]
    compatibility_threshold: float


@dataclass(frozen=True, slots=True)
class CouplingNodeUpdate:
    """Merged structural channels for one node."""

    node: Any
    theta_after: float | None
    vf_after: float | None
    dnfr_after: float | None


@dataclass(frozen=True, slots=True)
class CouplingEdgeProposal:
    """One deterministic new edge with a canonical weight."""

    left: Any
    right: Any
    weight: float


@dataclass(frozen=True, slots=True)
class CouplingStageProposal:
    """Fully merged and validated UM proposal."""

    targets: tuple[Any, ...]
    target_proposals: tuple[CouplingTargetProposal, ...]
    node_updates: tuple[CouplingNodeUpdate, ...]
    edges: tuple[CouplingEdgeProposal, ...]


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(
            f"Coupling {label} must be a finite real scalar",
            context={"operator": "Coupling", "field": label, "value": repr(value)},
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"Coupling {label} must be representable as a finite scalar",
            context={"operator": "Coupling", "field": label, "value": repr(value)},
        ) from exc
    if not math.isfinite(resolved):
        raise TNFRValueError(
            f"Coupling {label} must remain finite",
            context={"operator": "Coupling", "field": label, "value": repr(value)},
        )
    return resolved


def _nonnegative_real(value: Any, label: str) -> float:
    resolved = _finite_real(value, label)
    if resolved < 0.0:
        raise TNFRValueError(
            f"Coupling {label} must be nonnegative",
            context={"operator": "Coupling", "field": label, "value": resolved},
        )
    return resolved


def _unit_interval(value: Any, label: str) -> float:
    resolved = _finite_real(value, label)
    if not 0.0 <= resolved <= 1.0:
        raise TNFRValueError(
            f"Coupling {label} must lie in [0, 1]",
            context={"operator": "Coupling", "field": label, "value": resolved},
        )
    return resolved


def _raw_alias(
    graph: Any,
    node: Any,
    aliases: tuple[str, ...],
    default: Any = 0.0,
) -> Any:
    return get_attr(
        graph.nodes[node],
        aliases,
        default,
        strict=True,
        conv=lambda value: value,
    )


def _scalar_epi(graph: Any, node: Any, label: str) -> float:
    value = _raw_alias(graph, node, ALIAS_EPI)
    return require_real_scalar_epi(
        value,
        operator="Coupling",
        label=label,
    )


def compute_consensus_phase(phases: Sequence[float]) -> float:
    """Return the finite circular mean."""

    if not phases:
        return 0.0
    resolved = tuple(_finite_real(value, "phase input") for value in phases)
    cosine = math.fsum(math.cos(value) for value in resolved)
    sine = math.fsum(math.sin(value) for value in resolved)
    return math.atan2(sine, cosine)


def _candidate_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TNFRValueError(
            "Coupling UM_CANDIDATE_COUNT must be a nonnegative integer",
            context={"operator": "Coupling", "value": repr(value)},
        )
    resolved = int(value)
    if resolved < 0:
        raise TNFRValueError(
            "Coupling UM_CANDIDATE_COUNT must be a nonnegative integer",
            context={"operator": "Coupling", "value": resolved},
        )
    return resolved


def _candidate_ids(snapshot: Any) -> tuple[Any, ...]:
    configured = snapshot.graph.get("_node_sample")
    candidates = tuple(snapshot.nodes) if configured is None else tuple(configured)
    seen: set[Any] = set()
    for candidate in candidates:
        if candidate not in snapshot:
            raise TNFRValueError(
                "Coupling _node_sample contains an unknown node",
                context={"operator": "Coupling", "candidate": repr(candidate)},
            )
        if candidate in seen:
            raise TNFRValueError(
                "Coupling _node_sample must not repeat nodes",
                context={"operator": "Coupling", "candidate": repr(candidate)},
            )
        seen.add(candidate)
    return candidates


def _selected_candidate_ids(
    snapshot: Any,
    node: Any,
    *,
    target_phase: float,
    limit: int,
    mode: str,
    seed: int,
    node_offset: int,
    rank: Mapping[Any, int],
) -> tuple[Any, ...]:
    possible = tuple(
        candidate
        for candidate in _candidate_ids(snapshot)
        if candidate != node and not snapshot.has_edge(node, candidate)
    )
    try:
        selection = resolve_u3_phase_neighbors(
            snapshot.graph,
            target_phase,
            possible,
            phase_getter=lambda candidate: _raw_alias(
                snapshot, candidate, ALIAS_THETA, None
            ),
            operator_code="UM",
            require_compatible=False,
        )
    except U3PhaseGateError as exc:
        raise TNFRValueError(
            f"Coupling functional-link phase gate rejected a candidate: {exc}",
            context={
                "operator": "Coupling",
                "failed_condition": exc.failed_condition,
            },
        ) from exc
    candidates = selection.neighbors
    if limit == 0:
        return candidates
    if mode == "proximity":
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    abs(
                        angle_diff(
                            _finite_real(
                                _raw_alias(
                                    snapshot, candidate, ALIAS_THETA, None
                                ),
                                "candidate phase",
                            ),
                            target_phase,
                        )
                    ),
                    rank[candidate],
                ),
            )[:limit]
        )

    rng = make_rng(seed, node_offset)
    reservoir = list(candidates[:limit])
    for index, candidate in enumerate(candidates[limit:], start=limit):
        replacement = rng.randint(0, index)
        if replacement < limit:
            reservoir[replacement] = candidate
    # Unknown historical values intentionally retain reservoir order. Only
    # ``sample`` adds the established final shuffle; ``proximity`` is handled
    # above. This compatibility behavior is shared by direct and staged UM.
    if mode == "sample":
        rng.shuffle(reservoir)
    return tuple(reservoir)


def propose_coupling_target(
    snapshot: Any,
    node: Any,
    factors: Mapping[str, Any],
    *,
    resolved_seed: int | None,
    node_offset: int | None,
    rank: Mapping[Any, int],
) -> CouplingTargetProposal:
    """Build one target-local UM proposal without mutating the snapshot."""

    theta_push = _unit_interval(factors["UM_theta_push"], "UM_theta_push")

    try:
        selection = resolve_u3_phase_neighbors(
            snapshot.graph,
            _raw_alias(snapshot, node, ALIAS_THETA, None),
            tuple(snapshot.neighbors(node)),
            phase_getter=lambda neighbor: _raw_alias(
                snapshot, neighbor, ALIAS_THETA, None
            ),
            operator_code="UM",
        )
    except U3PhaseGateError as exc:
        raise TNFRValueError(
            f"Coupling phase gate rejected the operation: {exc}",
            context={
                "operator": "Coupling",
                "failed_condition": exc.failed_condition,
            },
        ) from exc

    bidirectional = bool(snapshot.graph.get("UM_BIDIRECTIONAL", True))
    inputs = (
        (selection.target_phase, *selection.phases)
        if bidirectional
        else selection.phases
    )
    consensus = compute_consensus_phase(inputs)
    proposed_target_phase = (
        selection.target_phase
        + theta_push * angle_diff(consensus, selection.target_phase)
    ) % math.tau
    phase_proposals = [
        CouplingPhaseProposal(
            source=node,
            node=node,
            theta_before=selection.target_phase,
            theta_proposed=_finite_real(
                proposed_target_phase, "target phase proposal"
            ),
        )
    ]
    if bidirectional:
        for neighbor, phase in zip(
            selection.neighbors, selection.phases, strict=True
        ):
            proposed = (
                phase + theta_push * angle_diff(consensus, phase)
            ) % math.tau
            phase_proposals.append(
                CouplingPhaseProposal(
                    source=node,
                    node=neighbor,
                    theta_before=phase,
                    theta_proposed=_finite_real(
                        proposed, "neighbor phase proposal"
                    ),
                )
            )

    write_vf = bool(snapshot.graph.get("UM_SYNC_VF", True))
    if write_vf:
        vf_sync = _unit_interval(factors["UM_vf_sync"], "UM_vf_sync")
        vf_before = _nonnegative_real(
            _raw_alias(snapshot, node, ALIAS_VF),
            "target structural frequency",
        )
        neighbor_vf = tuple(
            _nonnegative_real(
                _raw_alias(snapshot, neighbor, ALIAS_VF),
                "compatible-neighbor structural frequency",
            )
            for neighbor in selection.neighbors
        )
        vf_mean = math.fsum(neighbor_vf) / len(neighbor_vf)
        vf_after = _nonnegative_real(
            vf_before + vf_sync * (vf_mean - vf_before),
            "structural-frequency proposal",
        )
    else:
        vf_before = 0.0
        vf_after = 0.0

    write_dnfr = bool(snapshot.graph.get("UM_STABILIZE_DNFR", True))
    if write_dnfr:
        dnfr_reduction = _unit_interval(
            factors["UM_dnfr_reduction"], "UM_dnfr_reduction"
        )
        dnfr_before = _finite_real(
            _raw_alias(snapshot, node, ALIAS_DNFR), "target DeltaNFR"
        )
    else:
        dnfr_reduction = 0.0
        dnfr_before = 0.0

    functional_links = bool(
        snapshot.graph.get("UM_FUNCTIONAL_LINKS", True)
    )
    threshold = float(_UM_COMPAT_CANONICAL)
    link_candidates: list[CouplingLinkCandidate] = []
    if functional_links:
        threshold = _unit_interval(
            snapshot.graph.get(
                "UM_COMPAT_THRESHOLD",
                DEFAULTS.get("UM_COMPAT_THRESHOLD", _UM_COMPAT_CANONICAL),
            ),
            "UM_COMPAT_THRESHOLD",
        )
        if resolved_seed is None or node_offset is None:
            raise RuntimeError(
                "Coupling functional-link proposal lacks seed or node offset"
            )
        limit = _candidate_count(
            snapshot.graph.get("UM_CANDIDATE_COUNT", 0)
        )
        mode = str(
            snapshot.graph.get("UM_CANDIDATE_MODE", "sample")
        ).lower()
        epi_source = _scalar_epi(snapshot, node, "target EPI")
        si_source = _nonnegative_real(
            _raw_alias(snapshot, node, ALIAS_SI), "target sense index"
        )
        for candidate in _selected_candidate_ids(
            snapshot,
            node,
            target_phase=selection.target_phase,
            limit=limit,
            mode=mode,
            seed=resolved_seed,
            node_offset=node_offset,
            rank=rank,
        ):
            link_candidates.append(
                CouplingLinkCandidate(
                    source=node,
                    target=candidate,
                    epi_source=epi_source,
                    epi_target=_scalar_epi(
                        snapshot, candidate, "candidate EPI"
                    ),
                    si_source=si_source,
                    si_target=_nonnegative_real(
                        _raw_alias(snapshot, candidate, ALIAS_SI),
                        "candidate sense index",
                    ),
                )
            )

    return CouplingTargetProposal(
        node=node,
        glyph=Glyph.UM,
        compatible_neighbors=selection.neighbors,
        effective_phase_limit=selection.effective_limit,
        phase_proposals=tuple(phase_proposals),
        vf_before=vf_before,
        vf_after=vf_after,
        write_vf=write_vf,
        dnfr_before=dnfr_before,
        dnfr_reduction_factor=dnfr_reduction,
        write_dnfr=write_dnfr,
        link_candidates=tuple(link_candidates),
        compatibility_threshold=threshold,
    )


def _final_phase(
    node: Any,
    merged: Mapping[Any, float],
    snapshot_phase: Mapping[Any, float],
) -> float:
    return merged.get(node, snapshot_phase[node])


def _link_weight(
    candidate: CouplingLinkCandidate,
    phases: Mapping[Any, float],
) -> float:
    phase_strength = compute_phase_coupling_strength(
        phases[candidate.source], phases[candidate.target]
    )
    epi_scale = max(
        abs(candidate.epi_source),
        abs(candidate.epi_target),
        _EPI_SIMILARITY_EPSILON,
    )
    epi_distance = abs(
        candidate.epi_source / epi_scale
        - candidate.epi_target / epi_scale
    )
    epi_normalizer = (
        abs(candidate.epi_source) / epi_scale
        + abs(candidate.epi_target) / epi_scale
        + _EPI_SIMILARITY_EPSILON / epi_scale
    )
    epi_similarity = 1.0 - epi_distance / epi_normalizer
    si_similarity = 1.0 - abs(
        candidate.si_source - candidate.si_target
    )
    return _finite_real(
        0.5 * phase_strength
        + 0.25 * epi_similarity
        + 0.25 * si_similarity,
        "functional-link compatibility",
    )


def propose_coupling_stage(
    snapshot: Any,
    targets: Sequence[Any],
    factors: Mapping[str, Any],
    *,
    resolved_seed: int | None = None,
    node_offsets: Mapping[Any, int] | None = None,
) -> CouplingStageProposal:
    """Merge immutable target proposals with a deterministic engine policy."""

    targets_tuple = tuple(targets)
    if len(set(targets_tuple)) != len(targets_tuple):
        raise TNFRValueError(
            "Coupling stage targets must be unique",
            context={"operator": "Coupling"},
        )
    rank = {node: index for index, node in enumerate(snapshot.nodes)}
    for node in targets_tuple:
        if node not in rank:
            raise KeyError(node)

    offsets = {} if node_offsets is None else node_offsets
    target_proposals = tuple(
        propose_coupling_target(
            snapshot,
            node,
            factors,
            resolved_seed=resolved_seed,
            node_offset=offsets.get(node),
            rank=rank,
        )
        for node in targets_tuple
    )
    if tuple(proposal.node for proposal in target_proposals) != targets_tuple:
        raise RuntimeError("Coupling target proposal order changed")

    snapshot_phase = {
        node: _finite_real(
            _raw_alias(snapshot, node, ALIAS_THETA, None), "snapshot phase"
        )
        for node in snapshot.nodes
    }
    contributions: dict[Any, list[CouplingPhaseProposal]] = defaultdict(list)
    for proposal in target_proposals:
        for phase_proposal in proposal.phase_proposals:
            if phase_proposal.source != proposal.node:
                raise RuntimeError("Coupling phase proposal source changed")
            if phase_proposal.theta_before != snapshot_phase[
                phase_proposal.node
            ]:
                raise RuntimeError("Coupling phase proposal snapshot changed")
            contributions[phase_proposal.node].append(phase_proposal)

    merged_phase: dict[Any, float] = {}
    for node in sorted(contributions, key=rank.__getitem__):
        ordered = contributions[node]
        ordered.sort(key=lambda proposal: rank[proposal.source])
        displacement = math.fsum(
            angle_diff(
                proposal.theta_proposed, proposal.theta_before
            )
            for proposal in ordered
        ) / len(ordered)
        merged_phase[node] = _finite_real(
            (snapshot_phase[node] + displacement) % math.tau,
            "merged phase proposal",
        )

    final_phase = {
        node: _final_phase(node, merged_phase, snapshot_phase)
        for node in snapshot.nodes
    }

    # Existing target-neighbor relations used by the proposals must still obey
    # the same U3 limit after all overlapping phase writes are merged.
    for proposal in target_proposals:
        for neighbor in proposal.compatible_neighbors:
            separation = abs(
                angle_diff(
                    final_phase[proposal.node], final_phase[neighbor]
                )
            )
            if separation > proposal.effective_phase_limit:
                raise TNFRValueError(
                    "Coupling merged phase would violate the final U3 gate",
                    context={
                        "operator": "Coupling",
                        "node": repr(proposal.node),
                        "neighbor": repr(neighbor),
                        "separation": separation,
                        "phase_limit": proposal.effective_phase_limit,
                    },
                )

    updates_by_node: dict[Any, dict[str, float | None]] = {
        node: {"theta": phase, "vf": None, "dnfr": None}
        for node, phase in merged_phase.items()
    }
    for proposal in target_proposals:
        update = updates_by_node.setdefault(
            proposal.node, {"theta": None, "vf": None, "dnfr": None}
        )
        if proposal.write_vf:
            update["vf"] = proposal.vf_after
        if proposal.write_dnfr:
            strengths = tuple(
                compute_phase_coupling_strength(
                    final_phase[proposal.node], final_phase[neighbor]
                )
                for neighbor in proposal.compatible_neighbors
            )
            alignment = math.fsum(strengths) / len(strengths)
            multiplier = 1.0 - (
                proposal.dnfr_reduction_factor * alignment
            )
            if not 0.0 <= multiplier <= 1.0:
                raise RuntimeError(
                    "Coupling DeltaNFR reduction left the unit interval"
                )
            update["dnfr"] = _finite_real(
                proposal.dnfr_before * multiplier,
                "merged DeltaNFR proposal",
            )

    edge_weights: dict[Any, tuple[Any, Any, float]] = {}
    directed = bool(snapshot.is_directed())
    for proposal in target_proposals:
        for candidate in proposal.link_candidates:
            separation = abs(
                angle_diff(
                    final_phase[candidate.source],
                    final_phase[candidate.target],
                )
            )
            if separation > proposal.effective_phase_limit:
                continue
            weight = _link_weight(candidate, final_phase)
            if weight < proposal.compatibility_threshold:
                continue
            if not 0.0 <= weight <= 1.0:
                raise TNFRValueError(
                    "Coupling edge weight must lie in [0, 1]",
                    context={
                        "operator": "Coupling",
                        "weight": weight,
                    },
                )
            if snapshot.has_edge(candidate.source, candidate.target):
                continue
            if directed:
                key = (candidate.source, candidate.target)
                left, right = key
            else:
                left, right = sorted(
                    (candidate.source, candidate.target),
                    key=rank.__getitem__,
                )
                key = frozenset((left, right))
            previous = edge_weights.get(key)
            if previous is None or weight > previous[2]:
                edge_weights[key] = (left, right, weight)

    edges = tuple(
        CouplingEdgeProposal(left=left, right=right, weight=weight)
        for left, right, weight in sorted(
            edge_weights.values(),
            key=lambda item: (rank[item[0]], rank[item[1]]),
        )
    )
    node_updates = tuple(
        CouplingNodeUpdate(
            node=node,
            theta_after=updates_by_node[node]["theta"],
            vf_after=updates_by_node[node]["vf"],
            dnfr_after=updates_by_node[node]["dnfr"],
        )
        for node in sorted(updates_by_node, key=rank.__getitem__)
    )
    return CouplingStageProposal(
        targets=targets_tuple,
        target_proposals=target_proposals,
        node_updates=node_updates,
        edges=edges,
    )


__all__ = [
    "CouplingEdgeProposal",
    "CouplingLinkCandidate",
    "CouplingNodeUpdate",
    "CouplingPhaseProposal",
    "CouplingStageProposal",
    "CouplingTargetProposal",
    "compute_consensus_phase",
    "propose_coupling_stage",
    "propose_coupling_target",
]
