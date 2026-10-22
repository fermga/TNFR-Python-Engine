"""Synthetic SHA dispatch and primitive-reader controls, without source replay."""

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from math import inf, nextafter
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr, set_dnfr, set_vf
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.constants.canonical import SHA_VF_FACTOR
from tnfr.mathematics._rational_interval import I
from tnfr.operators.definitions import Coupling, Expansion, SelfOrganization, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.research import maintained_parent_sha_execution as owner
from tnfr.research import maintained_parent_sha_source as source_owner
from tnfr.research.local_capacity_history import CONTEXT, TIMES
from tnfr.research.maintained_parent_sha import predict_parent_sha
from tnfr.research.maintained_parent_sha_source import UmSuccessorAdmission
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.cache import DNFR_PREP_STATE_KEY, _graph_cache_manager


@dataclass(frozen=True)
class SyntheticStage:
    operator: str = "silence"
    glyph: str = "SHA"
    schedule: str = "two_phase_jacobi"
    nodes_processed: int = 1


def _clone(graph):
    return restore_recorded_graph_state(record_graph_state(graph)).graph


def _source():
    """Unrelated small-phase circulant support; no retained evidence is read."""
    graph = nx.Graph(
        _t=float(TIMES[-1]),
        GAMMA={"type": "none"},
        GLYPH_FACTORS={"VAL_scale": 128.0},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EDGE_AWARE_ENABLED=False,
        hierarchy={2: [owner.CHILD]},
    )
    nodes = tuple(range(20)) + (owner.CHILD,)
    for index, node in enumerate(nodes):
        graph.add_node(
            node,
            EPI=0.1 if index % 2 else 0.3,
            theta=0.1 * (index % 3),
            nu_f=1.3 if node == 2 else 1.0,
            glyph_history=[],
        )
    for offset in (1, 2):
        graph.add_edges_from(((i, (i + offset) % 20) for i in range(20)), weight=0.1)
    graph.add_edges_from(((i, i + 3) for i in range(10)), weight=0.1)
    graph.add_edges_from(((i, owner.CHILD) for i in (2, 8, 14)), weight=0.1)
    assert graph.number_of_edges() == 53
    graph.nodes[2].update(
        glyph_history=deque(("VAL", "THOL", "UM"), maxlen=7),
        _operator_step=3,
        _grammar_u2_debt=0,
        _grammar_prior_coherence=False,
        source_glyph="UM",
        epi_time_history=((float(TIMES[-2]), 0.25), (float(TIMES[-1]), 0.3)),
        sub_nodes=[owner.CHILD],
    )
    graph.nodes[owner.CHILD].update(parent_node=2, hierarchy_level=1, epi_history=[0.3])
    graph.nodes[0]["retained_marker"] = 1
    graph._last_operator_applied = "coupling"
    set_vf(graph, 2, 1.3)
    owner.refresh_pressure(graph)
    sequence = ValidatedSequence(
        (Expansion(), SelfOrganization(), Coupling(), Silence()), context=CONTEXT
    )
    previous = SimpleNamespace(
        chamber=SimpleNamespace(
            conditional_capacities=tuple(
                I(Q(graph.nodes[node]["nu_f"])) for node in nodes
            )
        )
    )
    return UmSuccessorAdmission(graph, None, previous, None, sequence)


@pytest.fixture(autouse=True)
def synthetic_source_boundary(monkeypatch):
    """Replace only source provenance; retain the real prospective predictor."""
    monkeypatch.setattr(source_owner, "validate_live_um_successor", lambda value: value)

    def forbidden(*args, **kwargs):
        pytest.fail("synthetic SHA controls must not read or execute retained evidence")

    monkeypatch.setattr(source_owner, "load_um_successor", forbidden)
    monkeypatch.setattr(owner, "execute_network_operator_stage", forbidden)


def _manual_successor(admitted):
    """Write independently specified outputs; do not invoke the SHA proposal."""
    raw = _clone(admitted.graph)
    before = Q(raw.nodes[2]["nu_f"])
    exact_product = before * Q(SHA_VF_FACTOR)
    set_vf(raw, 2, float(exact_product))
    raw.nodes[2].update(
        latent=True,
        latency_start_time="2030-01-02T03:04:08+00:00",
        preserved_epi=0.3,
        silence_duration=0.0,
        was_initial_on_silence=False,
        _operator_step=4,
        source_glyph="SHA",
    )
    raw.nodes[2]["glyph_history"].append("SHA")
    raw._last_operator_applied = "silence"
    endpoint = _clone(raw)
    owner.refresh_pressure(endpoint)
    endpoint.graph[owner.network_stage.STAGE_SCHEDULE_KEY] = owner.encode_result(
        SyntheticStage()
    )
    endpoint.graph[owner.network_stage.STAGE_CONTRACT_KEY] = (
        owner.stage_schedule_metadata("silence", observed_schedule="two_phase_jacobi")
    )
    return raw, endpoint


@pytest.fixture
def synthetic():
    admitted = _source()
    prediction = predict_parent_sha(admitted)
    raw, endpoint = _manual_successor(admitted)
    return admitted, prediction, raw, endpoint


def _check(synthetic):
    admitted, prediction, raw, endpoint = synthetic
    return owner.check_sha_endpoint(
        admitted, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )


def test_checker_rebuilds_exact_product_and_zero_work_from_primitive_state(synthetic):
    admitted, prediction, raw, endpoint = synthetic
    result = _check(synthetic)
    assert result["all_checks_passed"], result["checks"]
    ideal = Q(1.3) * Q(SHA_VF_FACTOR)
    represented = Q(float(ideal))
    assert represented != ideal
    assert prediction.ideal_parent_capacity == ideal
    assert prediction.represented_parent_capacity == represented
    assert Q(endpoint.nodes[2]["nu_f"]) == represented
    assert result["capacity_absolute_error"] == abs(represented - ideal)
    assert result["event_storage_jump"] == 0
    assert result["complete_word_executed"] is True
    assert result["um_motion_certificate_transferred"] is False
    assert result["numerical_u5_observation_available"] is False
    assert result["cross_birth_u6_observation_available"] is False
    assert raw.nodes[owner.CHILD] == admitted.graph.nodes[owner.CHILD]
    # The unchanged form energy is nonzero and has nonzero enclosure width.
    energy = Q(0.1) * (Q(0.3) - Q(0.1)) ** 2 / 2
    assert energy > 0 and I(energy).width > 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("latent", 1),
        ("silence_duration", 0),
        ("was_initial_on_silence", 0),
        ("preserved_epi", Q(0.3)),
        ("latency_start_time", "2030-01-02T03:04:09+00:00"),
    ],
)
def test_latency_requires_exact_declared_values_and_scalar_types(synthetic, key, value):
    for graph in synthetic[2:]:
        graph.nodes[2][key] = value
    result = _check(synthetic)
    assert result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["exact_latency_metadata"]
    assert not result["all_checks_passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda graph: graph.nodes[2].update(_operator_step=4.0),
        lambda graph: graph.nodes[2].update(_grammar_u2_debt=False),
        lambda graph: graph.nodes[2].update(_grammar_prior_coherence=0),
        lambda graph: graph.nodes[2].update(glyph_history=["VAL", "THOL", "UM", "SHA"]),
        lambda graph: graph.nodes[2].update(
            glyph_history=deque(("VAL", "THOL", "UM", "SHA"), maxlen=8)
        ),
    ],
)
def test_word_history_requires_exact_context_and_container(synthetic, mutation):
    for graph in synthetic[2:]:
        mutation(graph)
    result = _check(synthetic)
    assert not result["checks"]["actual_complete_parent_word"]
    assert not result["all_checks_passed"]


def test_child_cannot_acquire_parent_temporal_history(synthetic):
    admitted = synthetic[0]
    for graph in synthetic[2:]:
        graph.nodes[owner.CHILD]["epi_time_history"] = admitted.graph.nodes[2][
            "epi_time_history"
        ]
    result = _check(synthetic)
    assert not result["checks"]["child_temporal_baseline_unavailable"]
    assert not result["checks"]["declared_nodal_context_preserved"]


@pytest.mark.parametrize("channel", ["form", "phase", "conductance"])
def test_zero_work_requires_unchanged_storage_arguments(synthetic, channel):
    for graph in synthetic[2:]:
        if channel == "form":
            graph.nodes[0]["EPI"] = 0.4
        elif channel == "phase":
            graph.nodes[0]["theta"] = 0.125
        else:
            graph.edges[0, 1]["weight"] = 0.2
    result = _check(synthetic)
    assert not result["checks"]["storage_arguments_unchanged"]
    assert result["event_storage_jump"] is None
    assert not result["all_checks_passed"]


def test_raw_pressure_change_cannot_be_hidden_by_valid_refreshed_endpoint(synthetic):
    set_dnfr(synthetic[2], 2, 1.25)
    result = _check(synthetic)
    assert result["checks"]["refreshed_pressure_independently_bounded"]
    assert not result["checks"]["raw_pressure_preserved"]


def test_small_capacity_error_cannot_replace_exact_represented_reset(synthetic):
    for graph in synthetic[2:]:
        set_vf(graph, 2, nextafter(graph.nodes[2]["nu_f"], inf))
    result = _check(synthetic)
    assert result["checks"]["capacity_error_within_frozen_allowance"]
    assert not result["checks"]["exact_represented_capacity_reset"]
    assert not result["all_checks_passed"]


@pytest.mark.parametrize("count", [True, 0, 2, 1.0])
def test_stage_count_requires_exact_single_target_integer(synthetic, count):
    admitted, prediction, raw, endpoint = synthetic
    result = owner.check_sha_endpoint(
        admitted,
        raw,
        endpoint,
        prediction=prediction,
        stage=replace(SyntheticStage(), nodes_processed=count),
    )
    assert not result["checks"]["one_canonical_atomic_event"]
    assert not result["all_checks_passed"]


def test_raw_schedule_must_precede_final_schedule_metadata(synthetic):
    raw, endpoint = synthetic[2:]
    raw.graph[owner.network_stage.STAGE_SCHEDULE_KEY] = endpoint.graph[
        owner.network_stage.STAGE_SCHEDULE_KEY
    ]
    result = _check(synthetic)
    assert not result["checks"]["raw_schedule_precedes_final_metadata"]
    assert result["checks"]["final_schedule_metadata"]


def test_unchanged_neighbor_scalar_type_is_protected(synthetic):
    for graph in synthetic[2:]:
        graph.nodes[0]["retained_marker"] = True
    result = _check(synthetic)
    assert result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["declared_nodal_context_preserved"]


def test_dispatches_once_and_returns_primitives_before_assessment(monkeypatch):
    admitted = _source()
    prediction = predict_parent_sha(admitted)
    before = record_graph_state(admitted.graph)
    calls, durable = [], []

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        calls.append((operator.glyph.value, targets, sequence_context.index))
        compute_delta_nfr(graph)
        graph.graph["final_marker"] = True
        return SyntheticStage()

    def forbidden(*args, **kwargs):
        pytest.fail("response assessment must follow durable primitive persistence")

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    monkeypatch.setattr(owner, "check_sha_endpoint", forbidden)
    result = owner.evaluate_sha_event(
        admitted, frozen_before=before, prediction=prediction, on_raw=durable.append
    )
    assert calls == [("SHA", (2,), 3)]
    assert durable == [result["raw_after"]]
    assert set(result) == {
        "schema",
        "before",
        "raw_after",
        "endpoint",
        "prediction",
        "stage",
        "retained_source_receipts",
    }
    raw = restore_recorded_graph_state(result["raw_after"]).graph
    endpoint = restore_recorded_graph_state(result["endpoint"]).graph
    assert "final_marker" not in raw.graph
    assert endpoint.graph["final_marker"] is True


def test_changed_live_source_rejects_before_dispatch():
    admitted = _source()
    frozen = record_graph_state(admitted.graph)
    set_attr(admitted.graph.nodes[0], ALIAS_EPI, 0.4)
    with pytest.raises(ValueError, match="frozen SHA"):
        owner.evaluate_sha_event(admitted, frozen_before=frozen, prediction=None)


@pytest.mark.parametrize("handoffs", [0, 2])
def test_missing_or_repeated_handoff_rejects(monkeypatch, handoffs):
    admitted, durable = _source(), []

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        for _ in range(handoffs):
            compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    with pytest.raises(ValueError, match="handoff"):
        owner.evaluate_sha_event(
            admitted,
            frozen_before=record_graph_state(admitted.graph),
            prediction=None,
            on_raw=durable.append,
        )
    assert len(durable) == min(handoffs, 1)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda graph: graph.nodes[0].update(EPI=0.45),
        lambda graph: graph.nodes[0].update(nu_f=2.0),
        lambda graph: graph.nodes[0].update(theta=0.25),
        lambda graph: graph.nodes[2]["glyph_history"].append("IL"),
        lambda graph: graph.nodes[0].update(dnfr=0.0),
        lambda graph: graph.graph.update(_t=8.0),
        lambda graph: graph.edges[0, 1].update(weight=2.0),
    ],
)
def test_durable_raw_survives_forbidden_refresh_write(monkeypatch, mutation):
    admitted, durable = _source(), []

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    monkeypatch.setattr(owner, "refresh_pressure", mutation)
    with pytest.raises(ValueError, match="non-pressure"):
        owner.evaluate_sha_event(
            admitted,
            frozen_before=record_graph_state(admitted.graph),
            prediction=None,
            on_raw=durable.append,
        )
    assert len(durable) == 1
    raw = restore_recorded_graph_state(durable[0]).graph
    assert raw.graph["_t"] == float(TIMES[-1])
    assert tuple(raw.nodes[2]["glyph_history"]) == ("VAL", "THOL", "UM")


def test_unrelated_real_sha_stage_matches_complete_primitive_contract(monkeypatch):
    admitted = _source()
    before_admission = replace(admitted, graph=_clone(admitted.graph))
    prediction = predict_parent_sha(admitted)
    monkeypatch.setattr(
        owner, "execute_network_operator_stage", execute_network_operator_stage
    )
    result = owner.evaluate_sha_event(
        admitted,
        frozen_before=record_graph_state(admitted.graph),
        prediction=prediction,
    )
    raw, endpoint = (
        restore_recorded_graph_state(result[key]).graph
        for key in ("raw_after", "endpoint")
    )
    assessment = owner.check_sha_endpoint(
        before_admission, raw, endpoint, prediction=prediction, stage=result["stage"]
    )
    assert assessment["all_checks_passed"], assessment["checks"]
    assert get_attr(raw.nodes[2], ALIAS_DNFR) == get_attr(
        before_admission.graph.nodes[2], ALIAS_DNFR
    )


def test_warm_cache_owner_back_reference_allows_only_pressure_refresh(monkeypatch):
    admitted = _source()
    manager = _graph_cache_manager(admitted.graph.graph)
    manager.clear(DNFR_PREP_STATE_KEY)
    assert manager._graph_owner is admitted.graph.graph

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        set_dnfr(graph, 2, 1.25)
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    result = owner.evaluate_sha_event(
        admitted, frozen_before=record_graph_state(admitted.graph), prediction=None
    )
    raw = restore_recorded_graph_state(result["raw_after"]).graph
    endpoint = restore_recorded_graph_state(result["endpoint"]).graph
    assert raw.graph["_dnfrmax"] == 1.25
    assert endpoint.graph["_dnfrmax"] < 1.25
    assert admitted.graph.graph["_tnfr_cache_manager"] is manager
