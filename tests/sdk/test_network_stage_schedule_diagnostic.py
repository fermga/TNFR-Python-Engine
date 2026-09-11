"""Executable diagnostics for atomic EN/RA network-stage semantics."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.constants.operational import ACTIVE_EMISSION_THRESHOLD
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators.network_analysis.source_detection import detect_emission_sources
from tnfr.operators.definitions import Reception, Resonance
from tnfr.operators.network_stage import (
    GraphTransactionSnapshot,
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_neighbor_stage,
)
from tnfr.operators.strategies.gpu_strategies import _apply_canonical_block
from tnfr.physics.reception_realization import certify_reception_epi_realization
from tnfr.physics.resonance_realization import certify_resonance_epi_realization
from tnfr.sdk.simple import _run_network_sequence
from tnfr.types import real_scalar_epi, serialize_bepi


def _path(order: tuple[int, ...]) -> nx.Graph:
    graph = nx.Graph()
    epi = {0: 0.0, 1: 0.2, 2: 0.9}
    phase = {0: 0.0, 1: 0.2, 2: 0.4}
    for node in order:
        graph.add_node(
            node,
            EPI=epi[node],
            nu_f=1.0,
            delta_nfr=0.1,
            phase=phase[node],
            Si=0.8,
            EPI_kind="wave",
            glyph_history=["AL", "IL"],
        )
    graph.add_edges_from(((0, 1), (1, 2)))
    graph.graph["GLYPH_FACTORS"] = {
        "EN_mix": 0.25,
        "RA_epi_diff": 0.25,
        "RA_vf_amplification": 0.25,
        "RA_phase_coupling": 0.5,
    }
    return graph


def _state(graph: nx.Graph) -> dict[int, tuple[float, float, float]]:
    return {
        node: (
            float(real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))),
            float(get_attr(graph.nodes[node], ALIAS_VF)),
            float(get_attr(graph.nodes[node], ALIAS_THETA)),
        )
        for node in sorted(graph)
    }


def _observable_snapshot(graph: nx.Graph) -> tuple[dict, tuple, dict, tuple]:
    return (
        deepcopy({node: dict(data) for node, data in graph.nodes(data=True)}),
        deepcopy(tuple(graph.edges(data=True))),
        deepcopy(dict(graph.graph)),
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (
            "reception",
            {
                0: (0.05, 1.0, 0.0),
                1: (0.2625, 1.0, 0.2),
                2: (0.725, 1.0, 0.4),
            },
        ),
        (
            "resonance",
            {
                0: (0.05, 1.25, 0.1),
                1: (0.2625, 1.25, 0.2),
                2: (0.725, 1.25, 0.3),
            },
        ),
    ],
)
def test_sdk_neighbor_stage_is_insertion_order_invariant_and_matches_jacobi(
    operator: str,
    expected: dict[int, tuple[float, float, float]],
) -> None:
    outputs = []
    for order in ((0, 1, 2), (2, 1, 0)):
        graph = _path(order)
        _run_network_sequence(
            graph,
            [operator],
            validate=False,
            suppress_birth_warnings=True,
        )
        outputs.append(_state(graph))
        assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == TWO_PHASE_JACOBI
        contract = graph.graph[STAGE_CONTRACT_KEY]
        assert contract["observed_schedule"] == TWO_PHASE_JACOBI
        assert contract["schedule_matches_contract"] is True
        assert contract["executed_two_phase_contract_complete"] is True
        for node in graph:
            assert tuple(graph.nodes[node]["glyph_history"])[-1] == (
                "EN" if operator == "reception" else "RA"
            )

    for node, expected_state in expected.items():
        assert outputs[0][node] == pytest.approx(expected_state)
        assert outputs[1][node] == pytest.approx(expected_state)


def test_gpu_resonance_uses_the_same_two_phase_jacobi_stage() -> None:
    outputs = []
    for order in ((0, 1, 2), (2, 1, 0)):
        graph = _path(order)
        _apply_canonical_block(graph, "RA")
        outputs.append(_state(graph))
        assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == TWO_PHASE_JACOBI
        contract = graph.graph[STAGE_CONTRACT_KEY]
        assert contract["operator"] == "resonance"
        assert contract["glyph"] == "RA"
        assert contract["observed_schedule"] == TWO_PHASE_JACOBI
        assert contract["executed_two_phase_contract_complete"] is True

    expected = {
        0: (0.05, 1.25, 0.1),
        1: (0.2625, 1.25, 0.2),
        2: (0.725, 1.25, 0.3),
    }
    for node, expected_state in expected.items():
        assert outputs[0][node] == pytest.approx(expected_state)
        assert outputs[1][node] == pytest.approx(expected_state)


def test_sdk_stage_preflights_all_targets_before_any_commit() -> None:
    graph = _path((0, 1, 2))
    graph.nodes[2]["EPI"] = serialize_bepi(
        BEPIElement((-0.8, -0.7), (-0.8, -0.8), (0.0, 1.0))
    )
    before = _observable_snapshot(graph)

    with pytest.raises(TNFRValueError, match="uniform-real BEPI"):
        _run_network_sequence(graph, ["resonance"], validate=False)

    assert _observable_snapshot(graph) == before


def test_sdk_stage_rolls_back_a_failing_pressure_refresh() -> None:
    graph = _path((0, 1, 2))

    def rejected_refresh(subject: nx.Graph) -> None:
        subject.nodes[0]["delta_nfr"] = 99.0
        subject.graph["refresh_side_effect"] = True
        raise RuntimeError("pressure refresh rejected")

    graph.graph["compute_delta_nfr"] = rejected_refresh
    before = _observable_snapshot(graph)

    with pytest.raises(RuntimeError, match="pressure refresh rejected"):
        _run_network_sequence(graph, ["reception"], validate=False)

    assert _observable_snapshot(graph) == before


def test_resonance_stage_preserves_optional_telemetry_channels() -> None:
    graph = _path((0, 1, 2))
    graph.graph.update(
        COLLECT_OPERATOR_METRICS=True,
        COLLECT_RA_METRICS=True,
        TRACK_NETWORK_COHERENCE=True,
    )

    _run_network_sequence(graph, ["resonance"], validate=False)

    assert len(graph.graph["operator_metrics"]) == 3
    assert len(graph.graph["ra_metrics"]) == 3
    assert len(graph.graph["_ra_c_tracking"]) == 3
    assert {item["node"] for item in graph.graph["_ra_c_tracking"]} == {0, 1, 2}
    for node, metrics in zip(graph, graph.graph["ra_metrics"], strict=True):
        assert metrics["operator"] == "RA"
        assert metrics["epi_after"] == pytest.approx(
            real_scalar_epi(graph.nodes[node]["EPI"])
        )
        assert graph.nodes[node]["glyph_history"][-1] == "RA"


def test_resonance_structural_state_is_target_order_invariant_with_telemetry() -> None:
    class RecordingMonitor:
        def __init__(self) -> None:
            self.events: list[tuple[str, int]] = []

        def before_operator(self, graph, node) -> None:
            self.events.append(("before", node))

        def after_operator(self, graph, node, operator) -> None:
            self.events.append(("after", node))

    graphs = []
    orders = ((0, 1, 2), (2, 1, 0))
    for order in orders:
        graph = _path((0, 1, 2))
        monitor = RecordingMonitor()
        graph.graph.update(
            COLLECT_OPERATOR_METRICS=True,
            COLLECT_RA_METRICS=True,
            TRACK_NETWORK_COHERENCE=True,
            integrity_monitor=monitor,
        )

        execute_neighbor_stage(graph, Resonance(), order)
        graphs.append((graph, monitor))

    forward, forward_monitor = graphs[0]
    reverse, reverse_monitor = graphs[1]
    assert _state(forward) == _state(reverse)
    assert {
        node: tuple(forward.nodes[node]["glyph_history"]) for node in forward
    } == {
        node: tuple(reverse.nodes[node]["glyph_history"]) for node in reverse
    }

    assert [item["epi_before"] for item in forward.graph["ra_metrics"]] == [
        0.0,
        0.2,
        0.9,
    ]
    assert [item["epi_before"] for item in reverse.graph["ra_metrics"]] == [
        0.9,
        0.2,
        0.0,
    ]
    assert [item["node"] for item in forward.graph["_ra_c_tracking"]] == [0, 1, 2]
    assert [item["node"] for item in reverse.graph["_ra_c_tracking"]] == [2, 1, 0]
    assert forward_monitor.events == [
        (phase, node) for node in orders[0] for phase in ("before", "after")
    ]
    assert reverse_monitor.events == [
        (phase, node) for node in orders[1] for phase in ("before", "after")
    ]

    for graph, _monitor in graphs:
        contract = graph.graph[STAGE_CONTRACT_KEY]
        assert contract["structural_state_target_order_invariant"] is True
        assert "ordered lifecycle, telemetry and monitor streams" in contract[
            "structural_state_target_order_scope"
        ]
        assert contract["two_phase_contract_complete"] is True
        assert contract["executed_two_phase_contract_complete"] is True
        assert contract["relabeling_equivariant"] is None


def test_active_emission_threshold_is_centralized_and_matches_detection() -> None:
    graph = nx.path_graph(3)
    graph.nodes[0].update(EPI=0.1, nu_f=1.0, theta=0.0)
    graph.nodes[1].update(
        EPI=ACTIVE_EMISSION_THRESHOLD,
        nu_f=1.0,
        theta=0.0,
    )
    graph.nodes[2].update(
        EPI=ACTIVE_EMISSION_THRESHOLD - 0.01,
        nu_f=1.0,
        theta=0.0,
    )

    assert ACTIVE_EMISSION_THRESHOLD == 0.5
    sources = detect_emission_sources(graph, 0)
    assert [source for source, _phase, _strength in sources] == [1]


def test_neighbor_stage_rejects_duplicate_targets_before_writing() -> None:
    graph = _path((0, 1, 2))
    before = _observable_snapshot(graph)

    with pytest.raises(TNFRValueError, match="targets must be unique"):
        execute_neighbor_stage(graph, Reception(), [0, 0])

    assert _observable_snapshot(graph) == before


def test_resonance_stage_rejects_boolean_frequency_prewrite() -> None:
    graph = _path((0, 1, 2))
    graph.nodes[1][ALIAS_VF[0]] = True
    before = _observable_snapshot(graph)

    with pytest.raises(TNFRValueError, match="finite real scalar"):
        _run_network_sequence(graph, ["resonance"], validate=False)

    assert _observable_snapshot(graph) == before


def test_neighbor_stage_restores_object_cache_state_on_monitor_failure() -> None:
    class Cache:
        def __init__(self) -> None:
            self.value = 1

    class RejectingMonitor:
        def __init__(self, cache: Cache) -> None:
            self.cache = cache
            self.events: list[int] = []

        def before_operator(self, graph, node) -> None:
            self.events.append(node)

        def after_operator(self, graph, node, operator) -> None:
            self.cache.value = 9
            raise RuntimeError("reject stage")

    graph = _path((0, 1, 2))
    cache = Cache()
    monitor = RejectingMonitor(cache)
    graph.graph["custom_cache"] = cache
    graph.graph["integrity_monitor"] = monitor
    epi_before = _state(graph)

    with pytest.raises(RuntimeError, match="reject stage"):
        _run_network_sequence(graph, ["reception"], validate=False)

    assert graph.graph["custom_cache"] is cache
    assert cache.value == 1
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.cache is cache
    assert monitor.events == []
    assert _state(graph) == epi_before


def test_neighbor_stage_restores_supported_runtime_containers_and_graph_attrs() -> None:
    class SlotCache:
        __slots__ = ("value",)

        def __init__(self) -> None:
            self.value = 1

    class RuntimeGraph(nx.Graph):
        def __init__(self, token: str) -> None:
            super().__init__()
            self.token = token
            self.custom_state = {"value": 1}

    class CallableCache:
        def __init__(self) -> None:
            self.value = 1

        def __call__(self) -> int:
            return self.value

    class RejectingMonitor:
        def before_operator(self, graph, node) -> None:
            pass

        def after_operator(self, graph, node, operator) -> None:
            graph.graph["list_cache"].append(2)
            graph.graph["set_cache"].add(2)
            graph.graph["deque_cache"].append(2)
            graph.graph["array_cache"][0] = 9.0
            graph.graph["slot_cache"].value = 9
            graph.graph["callable_cache"].value = 9
            graph.custom_state["value"] = 9
            graph.transient_runtime_attr = "created during rejected stage"
            raise RuntimeError("reject supported runtime mutation")

    source = _path((0, 1, 2))
    graph = RuntimeGraph("kept")
    graph.graph.update(deepcopy(source.graph))
    graph.add_nodes_from(
        (node, deepcopy(data)) for node, data in source.nodes(data=True)
    )
    graph.add_edges_from(
        (left, right, deepcopy(data))
        for left, right, data in source.edges(data=True)
    )
    default_compute_delta_nfr(graph)
    cache_manager = graph.graph["_tnfr_cache_manager"]
    manager_storage = cache_manager._storage
    manager_owner = cache_manager._graph_owner
    list_cache = [1]
    set_cache = {1}
    deque_cache = deque((1,), maxlen=4)
    array_cache = np.asarray([1.0, 2.0])
    slot_cache = SlotCache()
    callable_cache = CallableCache()
    custom_state = graph.custom_state
    graph.graph.update(
        {
            "list_cache": list_cache,
            "set_cache": set_cache,
            "deque_cache": deque_cache,
            "array_cache": array_cache,
            "slot_cache": slot_cache,
            "callable_cache": callable_cache,
            "integrity_monitor": RejectingMonitor(),
        }
    )
    graph.graph["ordinary_metadata"] = {"runtime_alias": list_cache}

    with pytest.raises(RuntimeError, match="reject supported runtime mutation"):
        _run_network_sequence(graph, ["reception"], validate=False)

    assert graph.graph["list_cache"] is list_cache
    assert list_cache == [1]
    assert graph.graph["set_cache"] is set_cache
    assert set_cache == {1}
    assert graph.graph["deque_cache"] is deque_cache
    assert deque_cache == deque((1,), maxlen=4)
    assert graph.graph["array_cache"] is array_cache
    np.testing.assert_array_equal(array_cache, np.asarray([1.0, 2.0]))
    assert graph.graph["slot_cache"] is slot_cache
    assert slot_cache.value == 1
    assert graph.graph["callable_cache"] is callable_cache
    assert callable_cache.value == 1
    assert graph.graph["ordinary_metadata"]["runtime_alias"] is list_cache
    assert graph.graph["_tnfr_cache_manager"] is cache_manager
    assert cache_manager._storage is manager_storage
    assert cache_manager._storage_layer.storage is manager_storage
    assert cache_manager._graph_owner is manager_owner is graph.graph
    assert graph.custom_state is custom_state
    assert custom_state == {"value": 1}
    assert graph.token == "kept"
    assert not hasattr(graph, "transient_runtime_attr")


def test_neighbor_stage_rejects_unsupported_mutable_runtime_before_writes() -> None:
    graph = _path((0, 1, 2))
    opaque_cache = memoryview(bytearray(b"abc"))
    graph.graph["opaque_cache"] = opaque_cache
    before = _state(graph)

    with pytest.raises(TNFRValueError, match="unsupported mutable state"):
        _run_network_sequence(graph, ["reception"], validate=False)

    assert graph.graph["opaque_cache"] is opaque_cache
    assert opaque_cache.tobytes() == b"abc"
    assert _state(graph) == before


def test_detached_stage_supports_graph_subclasses_with_required_constructor() -> None:
    class NeedsArgument(nx.Graph):
        def __init__(self, token: str) -> None:
            super().__init__()
            self.token = token

    source = _path((0, 1, 2))
    graph = NeedsArgument("required")
    graph.graph.update(deepcopy(source.graph))
    graph.add_nodes_from(
        (node, deepcopy(data)) for node, data in source.nodes(data=True)
    )
    graph.add_edges_from(source.edges(data=True))

    _run_network_sequence(
        graph, ["reception"], validate=False, suppress_birth_warnings=True
    )

    assert graph.token == "required"
    assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == TWO_PHASE_JACOBI


def _graph_iteration_signature(graph: nx.Graph) -> tuple:
    adjacency = tuple(
        (node, tuple(graph.adj[node]))
        for node in graph
    )
    predecessors = (
        tuple((node, tuple(graph.pred[node])) for node in graph)
        if graph.is_directed()
        else None
    )
    edge_keys = (
        tuple(
            (node, neighbor, tuple(graph.adj[node][neighbor]))
            for node in graph
            for neighbor in graph.adj[node]
        )
        if graph.is_multigraph()
        else None
    )
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return tuple(graph), adjacency, predecessors, edge_keys, edges, tuple(graph.graph)


@pytest.mark.parametrize(
    "graph_type",
    (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
)
def test_transaction_snapshot_restores_exact_networkx_iteration_order(
    graph_type,
) -> None:
    graph = graph_type()
    graph.add_nodes_from((0, 1, 2))
    graph.graph.update(first=1, second=2)
    if graph.is_multigraph():
        graph.add_edge(2, 1, key="z", channel=1)
        graph.add_edge(2, 1, key="a", channel=2)
        graph.add_edge(0, 2, key="b", channel=3)
        graph.add_edge(0, 2, key="a", channel=4)
    else:
        graph.add_edge(2, 1, channel=1)
        graph.add_edge(0, 2, channel=2)
    before = _graph_iteration_signature(graph)
    snapshot = GraphTransactionSnapshot(graph)

    graph.remove_node(0)
    graph.add_node(9)
    graph.add_edge(9, 1)
    graph.graph.clear()
    graph.graph.update(second=9, first=8, transient=True)
    snapshot.restore(graph)

    assert _graph_iteration_signature(graph) == before


def test_failed_stage_restores_node_and_neighbor_order_after_topology_mutation() -> None:
    class TopologyMutatingMonitor:
        def before_operator(self, graph, node) -> None:
            pass

        def after_operator(self, graph, node, operator) -> None:
            graph.remove_node(0)
            raise RuntimeError("topology mutation rejected")

    graph = _path((0, 1, 2))
    graph.graph["integrity_monitor"] = TopologyMutatingMonitor()
    before = _graph_iteration_signature(graph)

    with pytest.raises(RuntimeError, match="topology mutation rejected"):
        _run_network_sequence(graph, ["reception"], validate=False)

    assert _graph_iteration_signature(graph) == before


def test_reception_all_target_stage_matches_each_snapshot_certificate() -> None:
    graph = _path((0, 1, 2))
    certificates = {
        node: certify_reception_epi_realization(
            graph, node, fixed_support_declared=True
        )
        for node in graph
    }

    _run_network_sequence(
        graph,
        ["reception"],
        validate=False,
        suppress_birth_warnings=True,
    )

    for node, certificate in certificates.items():
        assert real_scalar_epi(graph.nodes[node]["EPI"]) == pytest.approx(
            certificate.runtime_target_value
        )


def test_resonance_all_target_stage_matches_each_snapshot_certificate() -> None:
    graph = _path((0, 1, 2))
    certificates = {
        node: certify_resonance_epi_realization(
            graph, node, fixed_support_declared=True
        )
        for node in graph
    }

    _run_network_sequence(graph, ["resonance"], validate=False)

    for node, certificate in certificates.items():
        target_index = certificate.target_index
        assert real_scalar_epi(graph.nodes[node]["EPI"]) == pytest.approx(
            certificate.runtime_target_value
        )
        assert get_attr(graph.nodes[node], ALIAS_VF) == pytest.approx(
            certificate.frequency_after[target_index]
        )
        assert get_attr(graph.nodes[node], ALIAS_THETA) == pytest.approx(
            certificate.phase_after
        )
        assert graph.nodes[node]["EPI_kind"] == certificate.epi_kind_after
