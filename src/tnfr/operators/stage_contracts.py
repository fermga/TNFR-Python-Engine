"""Conservative execution contracts for all-target operator stages.

The canonical operator contracts describe physical effects.  This module
describes a separate implementation boundary: what the shared network executor
currently reads and writes when one operator is applied to every target.

The footprints are conservative unions over the public operator path used by
``run_network_sequence``, including optional telemetry, grammar bookkeeping,
configured branches and its final pressure refresh.  Direct per-node calls have
the separately reported rollback scope.  These contracts are not permissions to
change a schedule.  A stage may move from the established
operator-major Gauss--Seidel loop only after its proposal and merge law, complete
lifecycle commit, rollback, structural-state target-order behaviour and
relabeling scope have been established independently. Ordered audit and
telemetry streams may still retain the requested target order.

All thirteen canonical glyph stages meet the implemented two-phase boundary:

``immutable snapshot -> preflight all -> propose all -> validate all -> commit``.

Recursivity's stage proposal covers its node-level graph advisory only. It
deduplicates that advisory once per telemetry step and leaves structural
channels unchanged. The separate apply_network_remesh operation performs
explicit delayed network EPI mixing and is outside this glyph-stage contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .operator_contracts import contract_for, iter_contracts

__all__ = [
    "MergeLaw",
    "MergeLawStatus",
    "NetworkStageContract",
    "RollbackScope",
    "StageResource",
    "StageSchedule",
    "StructuralOverlap",
    "NETWORK_STAGE_CONTRACTS",
    "iter_stage_contracts",
    "remaining_stage_contract_gaps",
    "stage_contract_for",
    "stage_schedule_metadata",
    "verify_stage_contract_consistency",
]


class StageSchedule(str, Enum):
    """Network schedule implemented by the shared word executor."""

    TWO_PHASE_JACOBI = "two_phase_jacobi"
    OPERATOR_MAJOR_GAUSS_SEIDEL = "operator_major_gauss_seidel"


class StageResource(str, Enum):
    """Logical runtime resources appearing in stage read/write footprints."""

    TARGET_EPI = "target.EPI"
    TARGET_EPI_KIND = "target.EPI_kind"
    TARGET_NU_F = "target.nu_f"
    TARGET_THETA = "target.theta"
    TARGET_DELTA_NFR = "target.DeltaNFR"
    TARGET_D2_EPI = "target.d2EPI"
    TARGET_SI = "target.Si"
    TARGET_HISTORY = "target.operator_history"
    TARGET_METADATA = "target.lifecycle_metadata"
    NEIGHBOR_EPI = "neighbors.EPI"
    NEIGHBOR_EPI_KIND = "neighbors.EPI_kind"
    NEIGHBOR_NU_F = "neighbors.nu_f"
    NEIGHBOR_THETA = "neighbors.theta"
    NEIGHBOR_DELTA_NFR = "neighbors.DeltaNFR"
    NEIGHBOR_METADATA = "neighbors.lifecycle_metadata"
    ALL_NODE_EPI = "all_nodes.EPI"
    ALL_NODE_NU_F = "all_nodes.nu_f"
    ALL_NODE_THETA = "all_nodes.theta"
    ALL_NODE_DELTA_NFR = "all_nodes.DeltaNFR"
    ALL_NODE_D_EPI = "all_nodes.dEPI"
    ALL_NODE_SI = "all_nodes.Si"
    NODE_SUPPORT = "graph.node_support"
    EDGE_SUPPORT = "graph.edge_support"
    EDGE_ATTRIBUTES = "graph.edge_attributes"
    HIERARCHY = "graph.hierarchy"
    GRAPH_CONFIGURATION = "graph.configuration"
    GRAPH_TELEMETRY = "graph.telemetry"
    GRAPH_RUNTIME = "graph.monitor_cache_runtime"
    PRESSURE_REFRESH = "callback.compute_delta_nfr_effects"
    ROLLBACK_SNAPSHOT = "runtime.rollback_snapshot_all_graph_storage"
    RNG_PROGRESS = "runtime.rng_progress"
    WALL_CLOCK = "wall_clock"


class StructuralOverlap(str, Enum):
    """How one target's structural action can interact with another target."""

    DISJOINT_TARGET_WRITES = "disjoint_target_writes"
    NEIGHBOR_READ_AFTER_TARGET_WRITE = "neighbor_read_after_target_write"
    OVERLAPPING_NODE_WRITES = "overlapping_node_writes"
    OVERLAPPING_NODE_AND_EDGE_WRITES = "overlapping_node_and_edge_writes"
    SUPPORT_AND_HIERARCHY_WRITES = "support_and_hierarchy_writes"
    GRAPH_ADVISORY_ONLY = "graph_advisory_only"


class MergeLaw(str, Enum):
    """Current or required cross-target merge rule."""

    IMMUTABLE_PROPOSAL_COMMIT = "immutable_proposal_commit"
    DISJOINT_STRUCTURAL_UNION = "disjoint_structural_union"
    SNAPSHOT_TARGET_MERGE_REQUIRED = "snapshot_target_merge_required"
    SNAPSHOT_DELTA_NFR_ADDITIVE_REDUCTION = (
        "snapshot_DeltaNFR_additive_reduction"
    )
    SNAPSHOT_PHASE_TOPOLOGY_MERGE = "snapshot_phase_topology_merge"
    SNAPSHOT_SUPPORT_HIERARCHY_MERGE = "snapshot_support_hierarchy_merge"
    SNAPSHOT_ADVISORY_DEDUPLICATION = "snapshot_advisory_deduplication"


class MergeLawStatus(str, Enum):
    """Evidence level of the complete stage merge, including lifecycle state."""

    IMPLEMENTED_AND_TESTED = "implemented_and_tested"
    STRUCTURAL_STATE_CANDIDATE = "structural_state_candidate"
    UNSPECIFIED = "unspecified"


class RollbackScope(str, Enum):
    """Rollback boundary owned by the shared CPU stage path."""

    STAGE = "complete_stage"
    TARGET = "single_target_only"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class NetworkStageContract:
    """Observed and proved boundary of one all-target operator stage.

    ``rollback_scope`` describes the shared ``run_network_sequence`` entry
    point; ``direct_call_rollback_scope`` describes a public single-node call.
    ``structural_state_target_order_invariant`` concerns only the committed
    primary structural channels named by its scope. Ordered histories, pattern
    detections, telemetry and monitor events are separate observable streams.
    ``relabeling_equivariant`` is an independent tri-state claim. ``None`` means
    that no result is established. The scope string names every exclusion; in
    particular, the opaque pressure-refresh callback is not silently absorbed
    into a structural-state invariance claim.
    """

    name: str
    english_name: str
    glyph: str
    read_set: frozenset[StageResource]
    write_set: frozenset[StageResource]
    current_schedule: StageSchedule
    structural_overlap: StructuralOverlap
    merge_law: MergeLaw
    merge_law_status: MergeLawStatus
    rollback_scope: RollbackScope
    direct_call_rollback_scope: RollbackScope
    structural_state_target_order_invariant: bool | None
    structural_state_target_order_scope: str
    relabeling_equivariant: bool | None
    relabeling_scope: str
    blockers: tuple[str, ...]

    @property
    def stage_atomic(self) -> bool:
        """Whether the shared executor owns complete rollback for this stage."""

        return self.rollback_scope is RollbackScope.STAGE

    @property
    def two_phase_contract_complete(self) -> bool:
        """Whether the atomic proposal/commit boundary is implemented.

        Completion requires target-order invariance of the committed structural
        state. It does not canonicalize ordered lifecycle or telemetry streams.
        """

        return (
            self.current_schedule is StageSchedule.TWO_PHASE_JACOBI
            and self.merge_law_status is MergeLawStatus.IMPLEMENTED_AND_TESTED
            and self.stage_atomic
            and self.structural_state_target_order_invariant is True
        )


@dataclass(frozen=True, slots=True)
class _StageSpec:
    read_set: frozenset[StageResource]
    write_set: frozenset[StageResource]
    current_schedule: StageSchedule
    structural_overlap: StructuralOverlap
    merge_law: MergeLaw
    merge_law_status: MergeLawStatus
    rollback_scope: RollbackScope
    direct_call_rollback_scope: RollbackScope
    structural_state_target_order_invariant: bool | None
    structural_state_target_order_scope: str
    relabeling_equivariant: bool | None
    relabeling_scope: str
    blockers: tuple[str, ...]


R = StageResource
_COMMON_READ = frozenset(
    {
        R.TARGET_EPI,
        R.TARGET_EPI_KIND,
        R.TARGET_NU_F,
        R.TARGET_THETA,
        R.TARGET_DELTA_NFR,
        R.TARGET_HISTORY,
        R.TARGET_METADATA,
        R.GRAPH_CONFIGURATION,
        R.GRAPH_TELEMETRY,
        R.GRAPH_RUNTIME,
        R.PRESSURE_REFRESH,
        R.ROLLBACK_SNAPSHOT,
    }
)
_COMMON_WRITE = frozenset(
    {
        R.TARGET_HISTORY,
        R.TARGET_METADATA,
        R.GRAPH_TELEMETRY,
        R.GRAPH_RUNTIME,
        R.ALL_NODE_DELTA_NFR,
        R.PRESSURE_REFRESH,
    }
)
_JACOBI = StageSchedule.TWO_PHASE_JACOBI
_DISJOINT = StructuralOverlap.DISJOINT_TARGET_WRITES
_UNPROVED_RELABELING = (
    "no complete all-target relabeling-equivariance certificate"
)

_STAGE_SPECS: dict[str, _StageSpec] = {
    "emission": _StageSpec(
        _COMMON_READ | {R.TARGET_EPI, R.WALL_CLOCK},
        _COMMON_WRITE | {R.TARGET_EPI},
        _JACOBI,
        _DISJOINT,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed EPI only under an immutable stage snapshot and one shared "
        "stage timestamp; ordered lifecycle, telemetry and monitor streams "
        "retain requested target order; cache tie identity and opaque "
        "pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "reception": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_EPI,
            R.TARGET_EPI_KIND,
            R.NEIGHBOR_EPI,
            R.NEIGHBOR_EPI_KIND,
            R.ALL_NODE_EPI,
            R.ALL_NODE_NU_F,
            R.ALL_NODE_THETA,
            R.NODE_SUPPORT,
            R.EDGE_SUPPORT,
        },
        _COMMON_WRITE | {R.TARGET_EPI, R.TARGET_EPI_KIND},
        _JACOBI,
        StructuralOverlap.NEIGHBOR_READ_AFTER_TARGET_WRITE,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed EPI and EPI_kind only under an immutable stage snapshot; "
        "ordered lifecycle, telemetry and monitor streams retain requested "
        "target order; opaque pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "coherence": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_EPI,
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.NEIGHBOR_THETA,
            R.ALL_NODE_DELTA_NFR,
            R.ALL_NODE_D_EPI,
            R.NODE_SUPPORT,
            R.EDGE_SUPPORT,
        },
        _COMMON_WRITE | {R.TARGET_THETA, R.TARGET_DELTA_NFR},
        _JACOBI,
        StructuralOverlap.NEIGHBOR_READ_AFTER_TARGET_WRITE,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed target pressure and phase come from one immutable stage "
        "snapshot; canonical stage/global and radius-local C(t), legacy dispersion "
        "telemetry and ordered lifecycle, telemetry and monitor streams "
        "retain requested "
        "target order; opaque pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "dissonance": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.NEIGHBOR_NU_F,
            R.NEIGHBOR_THETA,
            R.NEIGHBOR_DELTA_NFR,
            R.EDGE_SUPPORT,
            R.EDGE_ATTRIBUTES,
            R.RNG_PROGRESS,
        },
        _COMMON_WRITE
        | {
            R.TARGET_DELTA_NFR,
            R.NEIGHBOR_DELTA_NFR,
            R.NEIGHBOR_METADATA,
            R.GRAPH_CONFIGURATION,
            R.RNG_PROGRESS,
        },
        _JACOBI,
        StructuralOverlap.OVERLAPPING_NODE_WRITES,
        MergeLaw.SNAPSHOT_DELTA_NFR_ADDITIVE_REDUCTION,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.TARGET,
        True,
        "committed DeltaNFR and per-node RNG progress use immutable local OZ "
        "proposals plus snapshot-rank-ordered math.fsum reduction of incoming "
        "propagation magnitudes; a missing graph seed resolves only within the "
        "transaction and persists on success; the local magnitude postcondition "
        "remains "
        "separate from signed accumulated pressure; ordered lifecycle, telemetry "
        "and monitor streams, including propagation events, retain requested "
        "target order; cache state and pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "coupling": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_EPI,
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.NEIGHBOR_EPI,
            R.NEIGHBOR_NU_F,
            R.NEIGHBOR_THETA,
            R.ALL_NODE_EPI,
            R.ALL_NODE_NU_F,
            R.ALL_NODE_THETA,
            R.TARGET_SI,
            R.ALL_NODE_SI,
            R.NODE_SUPPORT,
            R.EDGE_SUPPORT,
            R.RNG_PROGRESS,
        },
        _COMMON_WRITE
        | {
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.NEIGHBOR_THETA,
            R.EDGE_SUPPORT,
            R.EDGE_ATTRIBUTES,
            R.GRAPH_CONFIGURATION,
        },
        _JACOBI,
        StructuralOverlap.OVERLAPPING_NODE_AND_EDGE_WRITES,
        MergeLaw.SNAPSHOT_PHASE_TOPOLOGY_MERGE,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed theta, optional target nu_f and DeltaNFR, and deterministic "
        "new edge support/weights under one immutable snapshot; overlapping "
        "shortest-arc phase proposals use the engine's snapshot-rank-ordered "
        "mean-displacement policy; sampled link support is target-order "
        "invariant for the same resolved seed; ordered lifecycle, telemetry "
        "and monitor streams retain requested target order; cache state and "
        "opaque pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "resonance": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_EPI,
            R.TARGET_EPI_KIND,
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.NEIGHBOR_EPI,
            R.NEIGHBOR_EPI_KIND,
            R.NEIGHBOR_THETA,
            R.EDGE_SUPPORT,
        },
        _COMMON_WRITE
        | {R.TARGET_EPI, R.TARGET_EPI_KIND, R.TARGET_NU_F, R.TARGET_THETA},
        _JACOBI,
        StructuralOverlap.NEIGHBOR_READ_AFTER_TARGET_WRITE,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed EPI, EPI_kind, nu_f and theta only under an immutable stage "
        "snapshot; ordered lifecycle, telemetry and monitor streams retain "
        "requested target order; opaque pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "silence": _StageSpec(
        _COMMON_READ | {R.TARGET_EPI, R.TARGET_NU_F, R.WALL_CLOCK},
        _COMMON_WRITE | {R.TARGET_NU_F},
        _JACOBI,
        _DISJOINT,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed nu_f only under an immutable stage snapshot and one shared "
        "stage timestamp; ordered lifecycle, telemetry and monitor streams "
        "retain requested target order; cache tie identity and opaque "
        "pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "expansion": _StageSpec(
        _COMMON_READ | {R.TARGET_EPI, R.TARGET_NU_F},
        _COMMON_WRITE | {R.TARGET_EPI, R.TARGET_NU_F},
        _JACOBI,
        _DISJOINT,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed EPI and nu_f only under an immutable stage snapshot; "
        "ordered lifecycle, telemetry and monitor streams retain requested "
        "target order; cache tie identity and opaque pressure-refresh callback "
        "excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "contraction": _StageSpec(
        _COMMON_READ
        | {R.TARGET_EPI, R.TARGET_NU_F, R.TARGET_DELTA_NFR},
        _COMMON_WRITE
        | {R.TARGET_EPI, R.TARGET_NU_F, R.TARGET_DELTA_NFR},
        _JACOBI,
        _DISJOINT,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed EPI, nu_f and DeltaNFR only under an immutable stage "
        "snapshot; ordered lifecycle, telemetry and monitor streams retain "
        "requested target order; cache tie identity and opaque pressure-refresh "
        "callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "self_organization": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_EPI,
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.NEIGHBOR_EPI,
            R.NEIGHBOR_THETA,
            R.ALL_NODE_EPI,
            R.ALL_NODE_NU_F,
            R.ALL_NODE_THETA,
            R.ALL_NODE_DELTA_NFR,
            R.NODE_SUPPORT,
            R.EDGE_SUPPORT,
            R.HIERARCHY,
        },
        _COMMON_WRITE
        | {
            R.TARGET_DELTA_NFR,
            R.TARGET_D2_EPI,
            R.NODE_SUPPORT,
            R.HIERARCHY,
        },
        _JACOBI,
        StructuralOverlap.SUPPORT_AND_HIERARCHY_WRITES,
        MergeLaw.SNAPSHOT_SUPPORT_HIERARCHY_MERGE,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.TARGET,
        True,
        "committed d2EPI, DeltaNFR, child nodes, sub_nodes, sub_epis and graph "
        "hierarchy come from one immutable stage snapshot; cross-parent child-ID "
        "collisions and structural commit order resolve in snapshot-node rank; "
        "ordered lifecycle, telemetry and monitor streams retain requested target "
        "order; cache tie identity and opaque pressure-refresh callback excluded",
        None,
        "generated child identifiers depend on parent string rendering and "
        "snapshot-node rank; no complete relabeling-equivariance certificate",
        (),
    ),
    "mutation": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_EPI,
            R.TARGET_EPI_KIND,
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.TARGET_D2_EPI,
        },
        _COMMON_WRITE | {R.TARGET_THETA, R.TARGET_D2_EPI},
        _JACOBI,
        _DISJOINT,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed theta and structural-acceleration telemetry under immutable "
        "per-target growth evidence and one stage snapshot; ordered lifecycle, "
        "telemetry and monitor streams, including histories, bifurcation events "
        "and metrics, retain requested target order; opaque pressure-refresh "
        "callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "transition": _StageSpec(
        _COMMON_READ
        | {
            R.TARGET_EPI,
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.RNG_PROGRESS,
            R.WALL_CLOCK,
        },
        _COMMON_WRITE
        | {
            R.TARGET_NU_F,
            R.TARGET_THETA,
            R.TARGET_DELTA_NFR,
            R.RNG_PROGRESS,
        },
        _JACOBI,
        _DISJOINT,
        MergeLaw.IMMUTABLE_PROPOSAL_COMMIT,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "committed nu_f, theta, DeltaNFR and per-node RNG progress under one "
        "immutable stage snapshot and shared latency time; ordered lifecycle, "
        "telemetry and monitor streams, including warnings and transition "
        "events, retain requested target order; cache state and opaque "
        "pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
    "recursivity": _StageSpec(
        _COMMON_READ,
        _COMMON_WRITE,
        _JACOBI,
        StructuralOverlap.GRAPH_ADVISORY_ONLY,
        MergeLaw.SNAPSHOT_ADVISORY_DEDUPLICATION,
        MergeLawStatus.IMPLEMENTED_AND_TESTED,
        RollbackScope.STAGE,
        RollbackScope.NONE,
        True,
        "primary structural channels and support remain unchanged by the "
        "immutable advisory; one graph event is deduplicated per telemetry "
        "step; ordered lifecycle, telemetry and monitor streams retain "
        "requested target order; pressure-refresh callback excluded",
        None,
        _UNPROVED_RELABELING,
        (),
    ),
}


def _materialize_contracts() -> Mapping[str, NetworkStageContract]:
    canonical = {contract.name: contract for contract in iter_contracts()}
    missing = set(canonical) - set(_STAGE_SPECS)
    extra = set(_STAGE_SPECS) - set(canonical)
    if missing or extra:
        raise AssertionError(
            f"stage-contract coverage drift: missing {missing}, extra {extra}"
        )
    return MappingProxyType(
        {
            name: NetworkStageContract(
                name=name,
                english_name=canonical[name].english_name,
                glyph=canonical[name].glyph,
                **{
                    field: getattr(spec, field)
                    for field in _StageSpec.__dataclass_fields__
                },
            )
            for name, spec in _STAGE_SPECS.items()
        }
    )


NETWORK_STAGE_CONTRACTS = _materialize_contracts()


def stage_contract_for(identifier: Any) -> NetworkStageContract:
    """Resolve an executable identifier, display/class name, or glyph."""

    token = identifier.value if isinstance(identifier, Enum) else identifier
    if not isinstance(token, str):
        token = getattr(identifier, "name", token)
    canonical = contract_for(token)
    return NETWORK_STAGE_CONTRACTS[canonical.name]


def iter_stage_contracts() -> tuple[NetworkStageContract, ...]:
    """Return all 13 contracts in canonical operator-contract order."""

    return tuple(stage_contract_for(contract.name) for contract in iter_contracts())


def remaining_stage_contract_gaps() -> tuple[NetworkStageContract, ...]:
    """Return stages whose complete two-phase boundary is not established."""

    return tuple(
        contract
        for contract in iter_stage_contracts()
        if not contract.two_phase_contract_complete
    )


def stage_schedule_metadata(
    identifier: Any,
    *,
    observed_schedule: str | StageSchedule | None = None,
) -> dict[str, Any]:
    """Return JSON-safe diagnostic metadata for a shared-executor stage.

    ``observed_schedule`` lets runtime callers expose whether the recorded path
    matches the canonical schedule.  This matters when EN/RA grammar selection
    falls back to a heterogeneous sequential sweep: the requested operator owns
    a complete two-phase contract, while that particular execution does not use
    it.
    """

    contract = stage_contract_for(identifier)
    observed = (
        observed_schedule.value
        if isinstance(observed_schedule, StageSchedule)
        else observed_schedule
    )
    schedule_matches = (
        None if observed is None else observed == contract.current_schedule.value
    )
    return {
        "operator": contract.name,
        "glyph": contract.glyph,
        "observed_schedule": observed,
        "declared_schedule": contract.current_schedule.value,
        "schedule_matches_contract": schedule_matches,
        "read_set": sorted(resource.value for resource in contract.read_set),
        "write_set": sorted(resource.value for resource in contract.write_set),
        "structural_overlap": contract.structural_overlap.value,
        "merge_law": contract.merge_law.value,
        "merge_law_status": contract.merge_law_status.value,
        "rollback_scope": contract.rollback_scope.value,
        "direct_call_rollback_scope": contract.direct_call_rollback_scope.value,
        "stage_atomic": contract.stage_atomic,
        "structural_state_target_order_invariant": (
            contract.structural_state_target_order_invariant
        ),
        "structural_state_target_order_scope": (
            contract.structural_state_target_order_scope
        ),
        "relabeling_equivariant": contract.relabeling_equivariant,
        "relabeling_scope": contract.relabeling_scope,
        "two_phase_contract_complete": contract.two_phase_contract_complete,
        "executed_two_phase_contract_complete": (
            None
            if schedule_matches is None
            else contract.two_phase_contract_complete and schedule_matches
        ),
        "promotion_blockers": list(contract.blockers),
        "footprint_scope": (
            "shared run_network_sequence stage; custom pressure-refresh callback "
            "effects are opaque and represented by callback.compute_delta_nfr_effects"
        ),
    }


def verify_stage_contract_consistency() -> None:
    """Raise ``AssertionError`` if stage metadata drifts from the 13 operators."""

    canonical = {contract.name: contract for contract in iter_contracts()}
    assert set(NETWORK_STAGE_CONTRACTS) == set(canonical)
    assert len(NETWORK_STAGE_CONTRACTS) == 13
    for name, stage in NETWORK_STAGE_CONTRACTS.items():
        physical = canonical[name]
        assert (stage.english_name, stage.glyph) == (
            physical.english_name,
            physical.glyph,
        )
        assert stage.read_set
        assert stage.write_set
        assert stage.rollback_scope is RollbackScope.STAGE
        if stage.merge_law_status is MergeLawStatus.IMPLEMENTED_AND_TESTED:
            assert stage.two_phase_contract_complete

    completed = {
        contract.glyph
        for contract in NETWORK_STAGE_CONTRACTS.values()
        if contract.two_phase_contract_complete
    }
    assert completed == {
        "AL",
        "EN",
        "IL",
        "OZ",
        "UM",
        "RA",
        "SHA",
        "VAL",
        "NUL",
        "THOL",
        "ZHIR",
        "NAV",
        "REMESH",
    }
    assert not remaining_stage_contract_gaps()


verify_stage_contract_consistency()
