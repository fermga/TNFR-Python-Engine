"""Adversarial tests for the transactional all-target Coupling stage."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import Coupling
from tnfr.operators.metrics_network import coupling_metrics
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
)
from tnfr.operators.stage_contracts import stage_contract_for
from tnfr.types import serialize_bepi
from tnfr.utils import angle_diff


def _graph(
    phases: tuple[float, ...] = (0.0, 0.4, 0.8),
    *,
    graph_type: type[nx.Graph] = nx.Graph,
    edges: tuple[tuple[int, int], ...] | None = None,
) -> nx.Graph:
    graph = graph_type()
    for node, phase in enumerate(phases):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: 0.5 + 0.1 * node,
                ALIAS_VF[0]: float(node + 1),
                ALIAS_THETA[0]: phase,
                ALIAS_DNFR[0]: (-1.0 if node % 2 else 1.0) * (node + 1),
                ALIAS_SI[0]: 0.8,
                "glyph_history": ["AL"],
            },
        )
    selected_edges = edges
    if selected_edges is None:
        selected_edges = tuple(
            (node, node + 1) for node in range(len(phases) - 1)
        )
    graph.add_edges_from(selected_edges)
    graph.graph["RANDOM_SEED"] = 17
    graph.graph["UM_FUNCTIONAL_LINKS"] = False
    return graph


def _node_state(graph: nx.Graph) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            node,
            get_attr(data, ALIAS_THETA, 0.0),
            get_attr(data, ALIAS_VF, 0.0),
            get_attr(data, ALIAS_DNFR, 0.0),
            get_attr(data, ALIAS_EPI, 0.0),
        )
        for node, data in graph.nodes(data=True)
    )


def _edge_state(graph: nx.Graph) -> tuple[Any, ...]:
    if graph.is_multigraph():
        return tuple(
            (left, right, key, deepcopy(dict(data)))
            for left, right, key, data in graph.edges(keys=True, data=True)
        )
    return tuple(
        (left, right, deepcopy(dict(data)))
        for left, right, data in graph.edges(data=True)
    )


def _structural_state(graph: nx.Graph) -> tuple[Any, ...]:
    return _node_state(graph), _edge_state(graph)


def test_reverse_target_order_merges_shared_neighbor_identically() -> None:
    forward = _graph()
    reverse = deepcopy(forward)

    first = execute_coupling_stage(forward, Coupling(), (0, 2))
    second = execute_coupling_stage(reverse, Coupling(), (2, 0))

    assert _structural_state(forward) == _structural_state(reverse)
    assert first.schedule == second.schedule == TWO_PHASE_JACOBI
    assert forward.graph[STAGE_SCHEDULE_KEY]["nodes_processed"] == 2
    assert forward.nodes[1]["glyph_history"] == ["AL"]
    assert stage_contract_for("UM").two_phase_contract_complete is True


@pytest.mark.parametrize(
    ("bidirectional", "sync_vf", "stabilize_dnfr", "functional_links"),
    [
        (False, False, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ],
)
def test_direct_single_target_matches_stage(
    bidirectional: bool,
    sync_vf: bool,
    stabilize_dnfr: bool,
    functional_links: bool,
) -> None:
    direct = _graph((0.0, 0.2, 0.4, 0.6))
    staged = deepcopy(direct)
    for graph in (direct, staged):
        graph.graph.update(
            UM_BIDIRECTIONAL=bidirectional,
            UM_SYNC_VF=sync_vf,
            UM_STABILIZE_DNFR=stabilize_dnfr,
            UM_FUNCTIONAL_LINKS=functional_links,
            UM_COMPAT_THRESHOLD=0.0,
            RANDOM_SEED=91,
        )

    apply_glyph(direct, 1, "UM")
    execute_coupling_stage(staged, Coupling(), (1,))

    assert _structural_state(direct) == _structural_state(staged)
    assert direct.nodes[1]["glyph_history"] == staged.nodes[1]["glyph_history"]


def test_functional_links_accept_uniform_serialized_bepi_coordinates() -> None:
    graph = _graph((0.0, 0.1, 0.2), edges=((0, 1),))
    for node, epi in enumerate((0.25, -0.5, 0.75)):
        graph.nodes[node][ALIAS_EPI[0]] = serialize_bepi(epi)
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        RANDOM_SEED=5,
    )

    apply_glyph(graph, 0, "UM")

    assert graph.has_edge(0, 2)
    for node, epi in enumerate((0.25, -0.5, 0.75)):
        assert graph.nodes[node][ALIAS_EPI[0]] == serialize_bepi(epi)


def test_extreme_finite_epi_similarity_remains_bounded() -> None:
    graph = _graph((0.0, 0.1, 0.2), edges=((0, 1),))
    graph.nodes[0][ALIAS_EPI[0]] = 1e308
    graph.nodes[2][ALIAS_EPI[0]] = -1e308
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        RANDOM_SEED=5,
    )

    execute_coupling_stage(graph, Coupling(), (0,))

    weight = graph.edges[0, 2]["weight"]
    assert math.isfinite(weight)
    assert 0.0 <= weight <= 1.0


def test_negative_candidate_sense_index_is_rejected_before_writes() -> None:
    graph = _graph((0.0, 0.1, 0.2), edges=((0, 1),))
    graph.nodes[2][ALIAS_SI[0]] = -0.1
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        RANDOM_SEED=5,
    )
    before = _structural_state(graph)

    with pytest.raises(TNFRValueError, match="sense index must be nonnegative"):
        execute_coupling_stage(graph, Coupling(), (0,))

    assert _structural_state(graph) == before


def test_direct_phase_is_normalized_after_wrap() -> None:
    graph = _graph((math.tau - 0.001, 0.02))

    apply_glyph(graph, 0, "UM")

    for node in graph:
        phase = get_attr(graph.nodes[node], ALIAS_THETA, 0.0)
        assert 0.0 <= phase < math.tau


def test_direct_candidate_is_revalidated_against_final_phase() -> None:
    graph = _graph(
        (0.0, math.tau - 1.56, 1.56),
        edges=((0, 1),),
    )
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        RANDOM_SEED=3,
    )

    apply_glyph(graph, 0, "UM")

    assert not graph.has_edge(0, 2)
    final_separation = abs(
        angle_diff(
            get_attr(graph.nodes[0], ALIAS_THETA, 0.0),
            get_attr(graph.nodes[2], ALIAS_THETA, 0.0),
        )
    )
    assert final_separation > math.pi / 2.0


def test_snapshot_candidates_prevent_order_created_edges() -> None:
    forward = _graph((0.0, 0.0, 1.2, 2.0))
    reverse = deepcopy(forward)
    for graph in (forward, reverse):
        graph.graph.update(
            UM_FUNCTIONAL_LINKS=True,
            UM_COMPAT_THRESHOLD=0.0,
            UM_CANDIDATE_COUNT=0,
            RANDOM_SEED=12,
        )

    execute_coupling_stage(forward, Coupling(), tuple(forward))
    execute_coupling_stage(reverse, Coupling(), tuple(reversed(tuple(reverse))))

    assert _structural_state(forward) == _structural_state(reverse)
    assert forward.has_edge(0, 2)
    assert not forward.has_edge(0, 3)
    for left, right in forward.edges:
        separation = abs(
            angle_diff(
                get_attr(forward.nodes[left], ALIAS_THETA, 0.0),
                get_attr(forward.nodes[right], ALIAS_THETA, 0.0),
            )
        )
        assert separation <= math.pi / 2.0


@pytest.mark.parametrize(
    "graph_type",
    [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
)
def test_functional_edges_support_networkx_graph_families(
    graph_type: type[nx.Graph],
) -> None:
    graph = _graph(
        (0.0, 0.1, 0.2),
        graph_type=graph_type,
        edges=((0, 1),),
    )
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        RANDOM_SEED=8,
    )

    execute_coupling_stage(graph, Coupling(), (0,))

    assert graph.has_edge(0, 2)
    if graph.is_directed():
        assert not graph.has_edge(2, 0)
    if graph.is_multigraph():
        edge_data = graph.get_edge_data(0, 2)
        assert edge_data is not None
        assert len(edge_data) == 1
        weight = next(iter(edge_data.values()))["weight"]
    else:
        weight = graph.edges[0, 2]["weight"]
    assert 0.0 <= weight <= 1.0


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_directed_link_policy_can_add_the_missing_reverse_arc(
    graph_type: type[nx.Graph],
) -> None:
    graph = _graph(
        (0.0, 0.1, 0.2),
        graph_type=graph_type,
        edges=((0, 1), (2, 0)),
    )
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        RANDOM_SEED=8,
    )

    execute_coupling_stage(graph, Coupling(), (0,))

    assert graph.has_edge(2, 0)
    assert graph.has_edge(0, 2)


@pytest.mark.parametrize("graph_type", [nx.MultiGraph, nx.MultiDiGraph])
def test_existing_multiedge_pair_does_not_gain_an_implicit_parallel_edge(
    graph_type: type[nx.Graph],
) -> None:
    graph = _graph(
        (0.0, 0.1, 0.2),
        graph_type=graph_type,
        edges=((0, 1), (0, 2)),
    )
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        RANDOM_SEED=8,
    )
    before = graph.number_of_edges(0, 2)

    execute_coupling_stage(graph, Coupling(), (0,))

    assert graph.number_of_edges(0, 2) == before


def test_duplicate_edge_proposals_coalesce_deterministically() -> None:
    forward = _graph(
        (0.0, 0.1, 0.2, 0.3),
        graph_type=nx.MultiGraph,
        edges=((0, 1), (2, 3)),
    )
    reverse = deepcopy(forward)
    for graph in (forward, reverse):
        graph.graph.update(
            UM_FUNCTIONAL_LINKS=True,
            UM_COMPAT_THRESHOLD=0.0,
            RANDOM_SEED=42,
        )

    execute_coupling_stage(forward, Coupling(), (0, 2))
    execute_coupling_stage(reverse, Coupling(), (2, 0))

    assert _structural_state(forward) == _structural_state(reverse)
    assert forward.number_of_edges(0, 2) == 1


def test_negative_pressure_and_weight_metrics_use_magnitudes() -> None:
    graph = _graph((0.0, 0.1))
    graph.nodes[0][ALIAS_DNFR[0]] = -1.0
    graph.edges[0, 1]["weight"] = 0.7

    metrics = coupling_metrics(
        graph,
        0,
        theta_before=0.0,
        dnfr_before=-2.0,
    )

    assert metrics["dnfr_reduction"] == pytest.approx(1.0)
    assert metrics["dnfr_stabilization"] == pytest.approx(1.0)
    assert metrics["dnfr_reduction_pct"] == pytest.approx(50.0)
    assert metrics["coupling_strength_total"] == pytest.approx(0.7)


def test_multigraph_coupling_metric_sums_each_parallel_weight() -> None:
    graph = _graph((0.0, 0.1), graph_type=nx.MultiGraph, edges=())
    graph.add_edge(0, 1, weight=0.2)
    graph.add_edge(0, 1, coupling=0.3)

    metrics = coupling_metrics(graph, 0, theta_before=0.0)

    assert metrics["coupling_strength_total"] == pytest.approx(0.5)


def _rollback_state(graph: nx.Graph) -> tuple[Any, ...]:
    nodes = tuple(
        (
            node,
            repr(get_attr(data, ALIAS_THETA, 0.0)),
            repr(get_attr(data, ALIAS_VF, 0.0)),
            repr(get_attr(data, ALIAS_DNFR, 0.0)),
            tuple(data.get("glyph_history", ())),
        )
        for node, data in graph.nodes(data=True)
    )
    return (
        nodes,
        _edge_state(graph),
        graph.graph.get("RANDOM_SEED"),
        deepcopy(graph.graph.get(STAGE_SCHEDULE_KEY)),
        deepcopy(graph.graph.get("operator_metrics")),
        getattr(graph, "_last_operator_applied", None),
    )


def test_disabled_functional_links_do_not_resolve_seed_or_offset_cache() -> None:
    graph = _graph((0.0, 0.1, 0.2))
    graph.graph.update(
        RANDOM_SEED=None,
        UM_FUNCTIONAL_LINKS=False,
        UM_CANDIDATE_COUNT="unused-invalid-count",
        _node_sample=("unused-missing-node",),
    )

    execute_coupling_stage(graph, Coupling(), (0,))

    assert graph.graph["RANDOM_SEED"] is None
    assert "_node_list_cache" not in graph.graph
    assert "_node_cache" not in graph.graph


def test_global_seed_validation_still_applies_when_links_are_disabled() -> None:
    graph = _graph((0.0, 0.1))
    graph.graph.update(RANDOM_SEED="17", UM_FUNCTIONAL_LINKS=False)
    before = _rollback_state(graph)

    with pytest.raises(ValueError, match="RANDOM_SEED must be an integer"):
        execute_coupling_stage(graph, Coupling(), (0,))

    assert _rollback_state(graph) == before


def test_negative_compatible_capacity_is_rejected_before_direct_writes() -> None:
    graph = _graph((0.0, 0.1))
    graph.nodes[1][ALIAS_VF[0]] = -1.0
    before = _rollback_state(graph)

    with pytest.raises(TNFRValueError, match="must be nonnegative"):
        apply_glyph(graph, 0, "UM")

    assert _rollback_state(graph) == before


def test_disabled_capacity_sync_does_not_read_compatible_capacity() -> None:
    graph = _graph((0.0, 0.1))
    graph.nodes[1][ALIAS_VF[0]] = -1.0
    graph.graph["UM_SYNC_VF"] = False

    apply_glyph(graph, 0, "UM")

    assert get_attr(graph.nodes[0], ALIAS_VF, 0.0) == 1.0


def test_direct_late_edge_failure_restores_nodes_topology_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr import node as node_module

    graph = _graph(
        (0.0, 0.1, 0.2, 0.3),
        edges=((0, 1),),
    )
    graph.graph.update(
        RANDOM_SEED=None,
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
    )
    before = _rollback_state(graph)
    original_add_edge = node_module.add_edge
    calls = 0

    def fail_on_second_edge(
        candidate: nx.Graph,
        left: Any,
        right: Any,
        weight: float,
        overwrite: bool = False,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("late direct edge failure")
        return original_add_edge(
            candidate,
            left,
            right,
            weight,
            overwrite,
        )

    monkeypatch.setattr(node_module, "add_edge", fail_on_second_edge)

    with pytest.raises(RuntimeError, match="late direct edge failure"):
        apply_glyph(graph, 0, "UM")

    assert calls == 2
    assert _rollback_state(graph) == before
    assert graph.nodes[0]["glyph_history"] == ["AL"]


def test_late_stage_callback_failure_restores_resolved_seed_and_links() -> None:
    graph = _graph(
        (0.0, 0.1, 0.2),
        edges=((0, 1),),
    )
    graph.graph.update(
        RANDOM_SEED=None,
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
    )
    before = _rollback_state(graph)

    def fail_refresh(candidate: nx.Graph) -> None:
        candidate.nodes[0][ALIAS_DNFR[0]] = 999.0
        raise RuntimeError("late pressure refresh failure")

    with pytest.raises(RuntimeError, match="late pressure refresh failure"):
        execute_coupling_stage(
            graph,
            Coupling(),
            (0,),
            compute_delta_nfr=fail_refresh,
        )

    assert _rollback_state(graph) == before
