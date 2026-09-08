"""Immutable all-target proposal, reduction and lifecycle tests for OZ."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators._dissonance_stage_kernel import propose_dissonance_stage
from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_dissonance_stage,
)
from tnfr.operators.word_execution import run_network_sequence


def _context():
    return ValidatedSequence(
        [Emission(), Dissonance(), Coherence(), Silence()]
    ).step(1)


def _add_state(graph: nx.Graph, node: int, pressure: float) -> None:
    graph.add_node(
        node,
        **{
            ALIAS_EPI[0]: 0.5,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: pressure,
            ALIAS_THETA[0]: 0.0,
            "glyph_history": ["AL"],
            "source_glyph": "AL",
        },
    )


def _graph(graph_type: type[nx.Graph] = nx.Graph) -> nx.Graph:
    graph = graph_type(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=19,
        OZ_MIN_PROPAGATION=0.0,
    )
    for node, pressure in enumerate((0.2, 0.3, 0.4)):
        _add_state(graph, node, pressure)
    if graph.is_directed():
        graph.add_edge(0, 1, weight=0.6)
        graph.add_edge(2, 1, weight=0.8)
        graph.add_edge(1, 0, weight=0.5)
    else:
        graph.add_edge(0, 1, weight=0.6)
        graph.add_edge(1, 2, weight=0.8)
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=0.4)
        graph.add_edge(2, 1, weight=0.2)
    return graph


def _pressures(graph: nx.Graph) -> dict[int, float]:
    return {
        node: float(get_attr(graph.nodes[node], ALIAS_DNFR))
        for node in graph
    }


def _progress(graph: nx.Graph) -> dict[int, Any]:
    return {
        node: deepcopy(graph.nodes[node].get("_rng_jitter_progress"))
        for node in graph
    }


def _snapshot(graph: nx.Graph) -> tuple[Any, ...]:
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return (
        deepcopy(dict(graph.graph)),
        {node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)},
        deepcopy(edges),
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_overlapping_pressure_reduction_is_target_order_invariant(graph_type):
    forward = _graph(graph_type)
    reverse = _graph(graph_type)

    execute_dissonance_stage(
        forward, Dissonance(), (0, 1, 2), sequence_context=_context()
    )
    execute_dissonance_stage(
        reverse, Dissonance(), (2, 1, 0), sequence_context=_context()
    )

    assert _pressures(reverse) == _pressures(forward)
    assert {
        node: tuple(forward.nodes[node]["glyph_history"]) for node in forward
    } == {
        node: tuple(reverse.nodes[node]["glyph_history"]) for node in reverse
    }
    assert forward.graph[STAGE_SCHEDULE_KEY]["schedule"] == TWO_PHASE_JACOBI
    contract = forward.graph[STAGE_CONTRACT_KEY]
    assert contract["schedule_matches_contract"] is True
    assert contract["structural_state_target_order_invariant"] is True


def test_single_target_stage_matches_direct_local_and_propagated_structure():
    direct = nx.DiGraph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=23,
        OZ_MIN_PROPAGATION=0.0,
    )
    staged = nx.DiGraph(**deepcopy(direct.graph))
    for graph in (direct, staged):
        _add_state(graph, 0, 0.2)
        _add_state(graph, 1, 0.1)
        graph.add_edge(0, 1, weight=0.5)

    Dissonance()(direct, 0, sequence_context=_context())
    execute_dissonance_stage(
        staged, Dissonance(), (0,), sequence_context=_context()
    )

    assert _pressures(staged) == pytest.approx(_pressures(direct))
    for node in direct:
        assert dict(staged.nodes[node]) == dict(direct.nodes[node])
    assert staged.graph["_oz_propagation_events"] == direct.graph[
        "_oz_propagation_events"
    ]


def test_noise_streams_and_reduced_pressure_are_target_order_invariant():
    graphs = []
    for order in ((0, 1, 2), (2, 1, 0)):
        graph = _graph(nx.DiGraph)
        graph.graph.update(OZ_NOISE_MODE=True, OZ_SIGMA=0.25)
        execute_dissonance_stage(
            graph, Dissonance(), order, sequence_context=_context()
        )
        graphs.append(graph)

    assert _pressures(graphs[1]) == _pressures(graphs[0])
    assert _progress(graphs[1]) == _progress(graphs[0])
    assert all(
        graph.nodes[node]["_rng_jitter_progress"]["draws"] == 1
        for graph in graphs
        for node in graph
    )


def test_local_oz_contract_is_separate_from_signed_incoming_cancellation():
    graph = nx.DiGraph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=29,
        OZ_MIN_PROPAGATION=0.0,
        COLLECT_OPERATOR_METRICS=True,
    )
    _add_state(graph, 0, 0.3)
    _add_state(graph, 1, -0.2)
    graph.add_edge(0, 1, weight=1.0)

    proposal = propose_dissonance_stage(
        graph, (0, 1), propagate=True
    )
    local = next(item for item in proposal.target_proposals if item.node == 1)
    merged = next(item for item in proposal.pressure_updates if item.node == 1)

    assert local.dnfr_local_after == pytest.approx(-0.4)
    assert local.local_contract_satisfied is True
    assert merged.incoming_total == pytest.approx(0.3)
    assert merged.dnfr_after == pytest.approx(-0.1)
    assert abs(merged.dnfr_after) < abs(local.dnfr_before)

    execute_dissonance_stage(
        graph, Dissonance(), (0, 1), sequence_context=_context()
    )

    assert _pressures(graph)[1] == pytest.approx(-0.1)
    # Metrics retain the direct local-OZ observation boundary; the committed
    # field separately includes simultaneous incoming propagation.
    assert graph.graph["operator_metrics"][1]["dnfr_final"] == pytest.approx(
        -0.4
    )


def test_propagation_event_streams_retain_requested_source_order():
    forward = _graph(nx.DiGraph)
    reverse = _graph(nx.DiGraph)

    execute_dissonance_stage(
        forward, Dissonance(), (0, 2), sequence_context=_context()
    )
    execute_dissonance_stage(
        reverse, Dissonance(), (2, 0), sequence_context=_context()
    )

    assert _pressures(reverse) == _pressures(forward)
    assert [
        event["from_node"] for event in forward.nodes[1]["_oz_propagation"]
    ] == [0, 2]
    assert [
        event["from_node"] for event in reverse.nodes[1]["_oz_propagation"]
    ] == [2, 0]
    assert [
        event["source"] for event in forward.graph["_oz_propagation_events"]
    ] == [0, 2]
    assert [
        event["source"] for event in reverse.graph["_oz_propagation_events"]
    ] == [2, 0]


def test_strict_precondition_context_and_warnings_commit_in_target_order():
    graph = _graph(nx.DiGraph)
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        OZ_MIN_DEGREE=4,
    )

    with pytest.warns(UserWarning) as caught:
        execute_dissonance_stage(
            graph, Dissonance(), (2, 0, 1), sequence_context=_context()
        )

    messages = [str(item.message) for item in caught]
    assert "Node 2" in messages[0]
    assert "Node 0" in messages[1]
    assert "Node 1" in messages[2]
    for node in graph:
        context = graph.nodes[node]["_oz_precondition_context"]
        assert context["validation_passed"] is True
        assert context["dnfr"] == pytest.approx((0.2, 0.3, 0.4)[node])


def test_warning_promoted_to_error_restores_the_complete_stage():
    graph = _graph(nx.DiGraph)
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        OZ_MIN_DEGREE=4,
    )
    before = _snapshot(graph)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="low connectivity"):
            execute_dissonance_stage(
                graph, Dissonance(), (0, 1, 2), sequence_context=_context()
            )

    assert _snapshot(graph) == before


def test_late_invalid_neighbor_event_sink_rejects_before_any_live_write():
    graph = _graph(nx.DiGraph)
    graph.nodes[1]["_oz_propagation"] = ("invalid",)
    before = _snapshot(graph)

    with pytest.raises(TNFRValueError, match="_oz_propagation.*must be a list"):
        execute_dissonance_stage(
            graph, Dissonance(), (0, 1, 2), sequence_context=_context()
        )

    assert _snapshot(graph) == before


def test_late_pressure_refresh_failure_restores_complete_noisy_stage():
    graph = _graph(nx.MultiDiGraph)
    graph.graph.update(
        RANDOM_SEED=None,
        OZ_NOISE_MODE=True,
        OZ_SIGMA=0.25,
    )
    before = _snapshot(graph)

    def reject_refresh(subject: nx.Graph) -> None:
        subject.nodes[0][ALIAS_DNFR[0]] = 999.0
        subject.graph["refresh_side_effect"] = True
        raise RuntimeError("rejected OZ pressure refresh")

    with pytest.raises(RuntimeError, match="rejected OZ pressure refresh"):
        execute_dissonance_stage(
            graph,
            Dissonance(),
            (0, 1, 2),
            sequence_context=_context(),
            compute_delta_nfr=reject_refresh,
        )

    assert _snapshot(graph) == before


def test_shared_word_executor_routes_oz_through_the_jacobi_stage():
    graph = _graph(nx.DiGraph)
    observed = {}

    def capture_schedule(name: str) -> None:
        if name == "dissonance":
            observed.update(graph.graph[STAGE_SCHEDULE_KEY])

    run_network_sequence(
        graph,
        ["emission", "dissonance", "coherence", "silence"],
        validate=True,
        on_step=capture_schedule,
    )

    assert observed == {
        "operator": "dissonance",
        "glyph": "OZ",
        "schedule": TWO_PHASE_JACOBI,
        "nodes_processed": 3,
    }


def test_final_reduction_preserves_legacy_alias_and_pressure_cache():
    graph = nx.DiGraph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=31,
        OZ_MIN_PROPAGATION=0.0,
    )
    _add_state(graph, 0, 0.2)
    _add_state(graph, 1, 0.1)
    for node in graph:
        graph.nodes[node]["dnfr"] = graph.nodes[node].pop(ALIAS_DNFR[0])
    graph.add_edge(0, 1, weight=10.0)
    graph.graph.update(_dnfrmax=0.2, _dnfrmax_node=0)

    execute_dissonance_stage(
        graph, Dissonance(), (0,), sequence_context=_context()
    )

    assert ALIAS_DNFR[0] not in graph.nodes[0]
    assert ALIAS_DNFR[0] not in graph.nodes[1]
    assert graph.nodes[0]["dnfr"] == pytest.approx(0.4)
    assert graph.nodes[1]["dnfr"] == pytest.approx(2.1)
    assert graph.graph["_dnfrmax"] == pytest.approx(2.1)
    assert graph.graph["_dnfrmax_node"] == 1


def test_self_loop_local_and_propagated_overlap_matches_direct_execution():
    direct = nx.DiGraph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=37,
        OZ_MIN_PROPAGATION=0.0,
    )
    staged = nx.DiGraph(**deepcopy(direct.graph))
    for graph in (direct, staged):
        _add_state(graph, 0, 0.2)
        graph.add_edge(0, 0, weight=0.5)

    Dissonance()(direct, 0, sequence_context=_context())
    execute_dissonance_stage(
        staged, Dissonance(), (0,), sequence_context=_context()
    )

    assert _pressures(staged) == pytest.approx(_pressures(direct))
    assert staged.nodes[0]["_oz_propagation"] == direct.nodes[0][
        "_oz_propagation"
    ]


@pytest.mark.parametrize("invalid_pressure", [np.bool_(True), "0.2"])
def test_non_real_pressure_is_rejected_by_snapshot_domain(invalid_pressure):
    graph = _graph(nx.DiGraph)
    graph.nodes[0][ALIAS_DNFR[0]] = invalid_pressure
    before = _snapshot(graph)

    with pytest.raises(TNFRValueError, match="finite real scalar"):
        propose_dissonance_stage(graph, (0,), propagate=False)

    assert _snapshot(graph) == before


def test_missing_seed_is_materialized_once_with_per_node_progress():
    graph = _graph(nx.DiGraph)
    graph.graph.update(
        RANDOM_SEED=None,
        OZ_NOISE_MODE=True,
        OZ_SIGMA=0.25,
    )

    execute_dissonance_stage(
        graph, Dissonance(), (2, 0, 1), sequence_context=_context()
    )

    realized_seed = graph.graph["RANDOM_SEED"]
    assert isinstance(realized_seed, int)
    assert {
        graph.nodes[node]["_rng_jitter_progress"]["seed"] for node in graph
    } == {realized_seed}
    assert {
        graph.nodes[node]["_rng_jitter_progress"]["draws"] for node in graph
    } == {1}


def test_overflowing_incoming_reduction_rejects_before_live_write():
    graph = nx.DiGraph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=41,
        OZ_MIN_PROPAGATION=0.0,
    )
    _add_state(graph, 0, 1.0)
    _add_state(graph, 1, 0.1)
    _add_state(graph, 2, 1.0)
    graph.add_edge(0, 1, weight=1e308)
    graph.add_edge(2, 1, weight=1e308)
    before = _snapshot(graph)

    with pytest.raises(TNFRValueError, match="reduced pressure"):
        execute_dissonance_stage(
            graph, Dissonance(), (0, 2), sequence_context=_context()
        )

    assert _snapshot(graph) == before


def test_noncanonical_dissonance_override_uses_transactional_fallback(
    monkeypatch,
):
    graph = _graph(nx.DiGraph)
    original = Dissonance._execute
    calls = []

    def recording_execute(self, subject, node, **kwargs):
        calls.append(node)
        return original(self, subject, node, **kwargs)

    monkeypatch.setattr(Dissonance, "_execute", recording_execute)
    result = execute_dissonance_stage(
        graph, Dissonance(), (0, 1, 2), sequence_context=_context()
    )

    assert calls == [0, 1, 2]
    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == (
        OPERATOR_MAJOR_GAUSS_SEIDEL
    )
