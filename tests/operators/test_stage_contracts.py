"""Canonical coverage and conservative claims of network-stage contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_THETA
from tnfr.operators.definitions import Coherence
from tnfr.operators.network_stage import (
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    execute_operator_major_stage,
    execute_pointwise_stage,
)
from tnfr.operators.operator_contracts import iter_contracts
from tnfr.operators.stage_contracts import (
    NETWORK_STAGE_CONTRACTS,
    MergeLaw,
    MergeLawStatus,
    RollbackScope,
    StageResource,
    StageSchedule,
    StructuralOverlap,
    iter_stage_contracts,
    remaining_stage_contract_gaps,
    stage_contract_for,
    stage_schedule_metadata,
    verify_stage_contract_consistency,
)
from tnfr.types import Glyph


def test_registry_has_exact_identity_parity_with_canonical_contracts() -> None:
    canonical = iter_contracts()
    stages = iter_stage_contracts()

    assert len(stages) == 13
    assert set(NETWORK_STAGE_CONTRACTS) == {contract.name for contract in canonical}
    assert [
        (stage.name, stage.english_name, stage.glyph) for stage in stages
    ] == [
        (contract.name, contract.english_name, contract.glyph)
        for contract in canonical
    ]
    verify_stage_contract_consistency()


def test_registry_and_contract_values_are_immutable() -> None:
    contract = stage_contract_for("Emission")

    with pytest.raises(TypeError):
        NETWORK_STAGE_CONTRACTS["emission"] = contract  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        contract.glyph = "changed"  # type: ignore[misc]


def test_contract_lookup_accepts_all_canonical_identifier_forms() -> None:
    expected = NETWORK_STAGE_CONTRACTS["coherence"]

    assert stage_contract_for("coherence") is expected
    assert stage_contract_for("Coherence") is expected
    assert stage_contract_for("IL") is expected
    assert stage_contract_for(Glyph.IL) is expected
    assert stage_contract_for(Coherence()) is expected


def test_all_thirteen_operators_have_complete_two_phase_contracts() -> None:
    completed = {
        contract.glyph
        for contract in iter_stage_contracts()
        if contract.two_phase_contract_complete
    }
    gaps = remaining_stage_contract_gaps()

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
    assert not gaps
    for contract in iter_stage_contracts():
        expected = (
            StageSchedule.TWO_PHASE_JACOBI
            if contract.glyph in completed
            else StageSchedule.OPERATOR_MAJOR_GAUSS_SEIDEL
        )
        assert contract.current_schedule is expected


def test_shared_and_direct_call_rollback_scopes_are_not_conflated() -> None:
    contracts = iter_stage_contracts()

    assert all(contract.rollback_scope is RollbackScope.STAGE for contract in contracts)
    assert {
        contract.glyph
        for contract in contracts
        if contract.direct_call_rollback_scope is RollbackScope.TARGET
    } == {"OZ", "THOL"}
    assert all(
        contract.direct_call_rollback_scope is RollbackScope.NONE
        for contract in contracts
        if contract.glyph not in {"OZ", "THOL"}
    )


def test_pressure_refresh_is_an_explicit_opaque_stage_resource() -> None:
    for contract in iter_stage_contracts():
        assert StageResource.PRESSURE_REFRESH in contract.read_set
        assert StageResource.PRESSURE_REFRESH in contract.write_set
        assert StageResource.ROLLBACK_SNAPSHOT in contract.read_set
        assert StageResource.ALL_NODE_DELTA_NFR in contract.write_set

        metadata = stage_schedule_metadata(contract.glyph)
        assert metadata["operator"] == contract.name
        assert metadata["glyph"] == contract.glyph
        assert metadata["observed_schedule"] is None
        assert metadata["schedule_matches_contract"] is None
        assert metadata["executed_two_phase_contract_complete"] is None
        assert StageResource.PRESSURE_REFRESH.value in metadata["read_set"]
        assert StageResource.PRESSURE_REFRESH.value in metadata["write_set"]
        assert "custom pressure-refresh callback effects are opaque" in metadata[
            "footprint_scope"
        ]
        json.dumps(metadata)


def test_schedule_mismatch_does_not_claim_executed_two_phase_contract() -> None:
    metadata = stage_schedule_metadata(
        "RA",
        observed_schedule=StageSchedule.OPERATOR_MAJOR_GAUSS_SEIDEL,
    )

    assert metadata["operator"] == "resonance"
    assert metadata["glyph"] == "RA"
    assert metadata["observed_schedule"] == "operator_major_gauss_seidel"
    assert metadata["declared_schedule"] == "two_phase_jacobi"
    assert metadata["schedule_matches_contract"] is False
    assert metadata["two_phase_contract_complete"] is True
    assert metadata["executed_two_phase_contract_complete"] is False


def test_structural_target_order_and_relabeling_claims_remain_separate() -> None:
    for glyph in (
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
    ):
        contract = stage_contract_for(glyph)
        assert contract.structural_state_target_order_invariant is True
        assert contract.relabeling_equivariant is None
        assert "ordered lifecycle, telemetry and monitor streams" in (
            contract.structural_state_target_order_scope
        )
        assert "retain requested target order" in (
            contract.structural_state_target_order_scope
        )
        assert "pressure-refresh callback excluded" in (
            contract.structural_state_target_order_scope
        )


    thol = stage_contract_for("THOL")
    assert thol.relabeling_equivariant is None
    assert "snapshot-node rank" in thol.relabeling_scope


def test_cross_target_footprints_pin_current_merge_contracts() -> None:
    reception = stage_contract_for("EN")
    assert {
        StageResource.ALL_NODE_EPI,
        StageResource.ALL_NODE_NU_F,
        StageResource.ALL_NODE_THETA,
        StageResource.NODE_SUPPORT,
        StageResource.EDGE_SUPPORT,
    } <= reception.read_set

    coherence = stage_contract_for("IL")
    assert (
        coherence.structural_overlap
        is StructuralOverlap.NEIGHBOR_READ_AFTER_TARGET_WRITE
    )
    assert {
        StageResource.NEIGHBOR_THETA,
        StageResource.ALL_NODE_DELTA_NFR,
    } <= coherence.read_set

    dissonance = stage_contract_for("OZ")
    assert dissonance.structural_overlap is StructuralOverlap.OVERLAPPING_NODE_WRITES
    assert {
        StageResource.NEIGHBOR_DELTA_NFR,
        StageResource.NEIGHBOR_METADATA,
        StageResource.GRAPH_CONFIGURATION,
    } <= dissonance.write_set
    assert dissonance.current_schedule is StageSchedule.TWO_PHASE_JACOBI
    assert dissonance.merge_law is (
        MergeLaw.SNAPSHOT_DELTA_NFR_ADDITIVE_REDUCTION
    )
    assert dissonance.merge_law_status is (
        MergeLawStatus.IMPLEMENTED_AND_TESTED
    )

    coupling = stage_contract_for("UM")
    assert (
        coupling.structural_overlap
        is StructuralOverlap.OVERLAPPING_NODE_AND_EDGE_WRITES
    )
    assert {
        StageResource.NEIGHBOR_THETA,
        StageResource.EDGE_SUPPORT,
        StageResource.EDGE_ATTRIBUTES,
    } <= coupling.write_set
    assert {
        StageResource.TARGET_SI,
        StageResource.ALL_NODE_SI,
        StageResource.NODE_SUPPORT,
    } <= coupling.read_set
    assert StageResource.GRAPH_CONFIGURATION in coupling.write_set
    assert coupling.merge_law is MergeLaw.SNAPSHOT_PHASE_TOPOLOGY_MERGE

    thol = stage_contract_for("THOL")
    assert thol.structural_overlap is StructuralOverlap.SUPPORT_AND_HIERARCHY_WRITES
    assert {StageResource.NODE_SUPPORT, StageResource.HIERARCHY} <= thol.write_set
    assert {
        StageResource.ALL_NODE_EPI,
        StageResource.ALL_NODE_NU_F,
        StageResource.ALL_NODE_THETA,
        StageResource.ALL_NODE_DELTA_NFR,
    } <= thol.read_set
    assert thol.current_schedule is StageSchedule.TWO_PHASE_JACOBI
    assert thol.merge_law is MergeLaw.SNAPSHOT_SUPPORT_HIERARCHY_MERGE
    assert thol.merge_law_status is MergeLawStatus.IMPLEMENTED_AND_TESTED
    assert thol.structural_state_target_order_invariant is True

    recursivity = stage_contract_for("REMESH")
    assert (
        recursivity.structural_overlap
        is StructuralOverlap.GRAPH_ADVISORY_ONLY
    )
    assert recursivity.current_schedule is StageSchedule.TWO_PHASE_JACOBI
    assert (
        recursivity.merge_law
        is MergeLaw.SNAPSHOT_ADVISORY_DEDUPLICATION
    )
    assert (
        recursivity.merge_law_status
        is MergeLawStatus.IMPLEMENTED_AND_TESTED
    )
    assert recursivity.blockers == ()


def _coherence_path() -> nx.Graph:
    graph = nx.path_graph(3)
    theta = {0: 0.0, 1: 1.0, 2: 2.0}
    dnfr = {0: 0.8, 1: -0.4, 2: 0.2}
    for node in graph:
        graph.nodes[node].update(
            EPI=0.5,
            nu_f=1.0,
            theta=theta[node],
            DeltaNFR=dnfr[node],
            glyph_history=["AL", "OZ"],
        )
    return graph


def _phases(graph: nx.Graph) -> tuple[float, ...]:
    return tuple(
        float(get_attr(graph.nodes[node], ALIAS_THETA)) for node in sorted(graph)
    )


def test_coherence_canonical_stage_is_jacobi_and_target_order_invariant() -> None:
    forward = _coherence_path()
    reverse = _coherence_path()

    execute_pointwise_stage(forward, Coherence(), (0, 1, 2))
    execute_pointwise_stage(reverse, Coherence(), (2, 1, 0))

    assert _phases(forward) == pytest.approx((0.3, 1.0, 1.7))
    assert _phases(reverse) == pytest.approx(_phases(forward))
    schedule = forward.graph[STAGE_SCHEDULE_KEY]
    assert schedule == {
        "operator": "coherence",
        "glyph": "IL",
        "schedule": "two_phase_jacobi",
        "nodes_processed": 3,
    }
    diagnostic = forward.graph[STAGE_CONTRACT_KEY]
    assert diagnostic["declared_schedule"] == "two_phase_jacobi"
    assert diagnostic["schedule_matches_contract"] is True
    assert diagnostic["structural_state_target_order_invariant"] is True
    assert diagnostic["executed_two_phase_contract_complete"] is True


def test_explicit_legacy_coherence_executor_reports_schedule_mismatch() -> None:
    forward = _coherence_path()
    reverse = _coherence_path()

    execute_operator_major_stage(forward, Coherence(), (0, 1, 2))
    execute_operator_major_stage(reverse, Coherence(), (2, 1, 0))

    assert _phases(forward) == pytest.approx((0.3, 1.045, 1.7135))
    assert _phases(reverse) == pytest.approx((0.2865, 0.955, 1.7))
    assert _phases(forward) != pytest.approx(_phases(reverse))
    diagnostic = forward.graph[STAGE_CONTRACT_KEY]
    assert diagnostic["observed_schedule"] == "operator_major_gauss_seidel"
    assert diagnostic["declared_schedule"] == "two_phase_jacobi"
    assert diagnostic["schedule_matches_contract"] is False
    assert diagnostic["executed_two_phase_contract_complete"] is False
