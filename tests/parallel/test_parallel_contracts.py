"""Evidence contracts for the parallel and distributed computation facades."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_THETA, ALIAS_VF
from tnfr.metrics.sense_index import compute_Si
from tnfr.parallel import FractalPartitioner
from tnfr.parallel.distributed import (
    TNFRDistributedEngine,
    _compute_si_chunk,
    _distributed_compute_kwargs,
    _merge_si_chunks,
)
from tnfr.parallel.engine import TNFRParallelEngine


def _graph(graph_type=nx.Graph):
    graph = graph_type()
    graph.add_edge("a", "b", weight=2.0)
    graph.nodes["a"].update(nu_f=1.0, phase=0.1, delta_nfr=-0.2)
    graph.nodes["b"].update(nu_f=0.5, phase=0.3, delta_nfr=0.4)
    return graph


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_parallel_engine_rejects_invalid_worker_counts(value):
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        TNFRParallelEngine(max_workers=value)


def test_parallel_engine_rejects_invented_execution_mode():
    with pytest.raises(ValueError, match="execution_mode"):
        TNFRParallelEngine(execution_mode="gpu")


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_worker_recommendation_rejects_invalid_graph_size(value):
    with pytest.raises((TypeError, ValueError), match="nonnegative integer"):
        TNFRParallelEngine(max_workers=2).recommend_workers(value)


def test_parallel_si_is_a_nonmutating_node_mapping():
    graph = _graph()
    values = TNFRParallelEngine(max_workers=1).compute_si_parallel(graph)
    assert list(values) == list(graph)
    assert all(0.0 <= value <= 1.0 for value in values.values())
    assert all("Si" not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("backend", ["gpu", "", 1, True])
def test_distributed_engine_rejects_unknown_backend(backend):
    with pytest.raises(ValueError, match="backend must be one of"):
        TNFRDistributedEngine(backend=backend)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_distributed_engine_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        TNFRDistributedEngine(backend="multiprocessing").compute_si_distributed(
            _graph(), chunk_size=chunk_size
        )


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_chunk_worker_matches_global_si_without_reconstructing_graph_kind(graph_type):
    graph = _graph(graph_type)
    expected = compute_Si(graph, inplace=False, n_jobs=1)
    actual = _compute_si_chunk(list(graph), graph, {"inplace": False, "n_jobs": 1})
    assert actual == pytest.approx(expected)
    assert type(graph) is graph_type


def test_chunk_worker_propagates_metric_errors(monkeypatch):
    import tnfr.metrics.sense_index as sense_index

    def fail(*args, **kwargs):
        raise RuntimeError("measured failure")

    monkeypatch.setattr(sense_index, "compute_Si", fail)
    with pytest.raises(RuntimeError, match="measured failure"):
        _compute_si_chunk(["a"], _graph(), {"inplace": False, "n_jobs": 1})


@pytest.mark.parametrize(
    "kwargs",
    [{"inplace": True}, {"n_jobs": 2}, {"profile": {}}],
)
def test_remote_kwargs_reject_unobservable_or_nested_execution(kwargs):
    with pytest.raises(ValueError):
        _distributed_compute_kwargs(kwargs)


def test_merge_rejects_duplicate_missing_and_foreign_nodes():
    with pytest.raises(RuntimeError, match="duplicated"):
        _merge_si_chunks(["a"], ({"a": 0.2}, {"a": 0.3}))
    with pytest.raises(RuntimeError, match="omitted"):
        _merge_si_chunks(["a", "b"], ({"a": 0.2},))
    with pytest.raises(RuntimeError, match="foreign"):
        _merge_si_chunks(["a"], ({"a": 0.2, "b": 0.3},))


def test_local_fallback_reports_compute_path_without_claiming_observed_parallelism():
    result = TNFRDistributedEngine(backend="multiprocessing").compute_si_distributed(
        _graph(), chunk_size=1, n_jobs=1
    )
    assert result["backend"] == "compute_Si"
    assert result["selected_backend"] == "multiprocessing"
    assert result["parallel_execution_observed"] is None


def test_simulation_rejects_silently_ignored_operator_sequences():
    engine = TNFRDistributedEngine(backend="multiprocessing")
    with pytest.raises(NotImplementedError, match="atomic cross-partition"):
        engine.simulate_large_network(4, 0.5, [["AL", "SHA"]])


def test_simulation_seed_makes_generated_graph_statistics_reproducible():
    engine = TNFRDistributedEngine(backend="multiprocessing")
    first = engine.simulate_large_network(20, 0.2, [], chunk_size=5, seed=17)
    second = engine.simulate_large_network(20, 0.2, [], chunk_size=5, seed=17)
    assert first["network_stats"] == second["network_stats"]
def test_partitioner_exposes_affinity_name_and_legacy_alias():
    partitioner = FractalPartitioner(affinity_threshold=0.7)
    assert partitioner.affinity_threshold == pytest.approx(0.7)
    assert partitioner.coherence_threshold == pytest.approx(0.7)

    partitioner.coherence_threshold = 0.4
    assert partitioner.affinity_threshold == pytest.approx(0.4)

    with pytest.raises(ValueError, match="must agree"):
        FractalPartitioner(
            coherence_threshold=0.2,
            affinity_threshold=0.8,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf, True, "0.3"])
def test_partitioner_rejects_invalid_affinity_threshold(threshold):
    with pytest.raises(ValueError, match=r"finite real in \[0, 1\]"):
        FractalPartitioner(affinity_threshold=threshold)


def test_partitioner_affinity_alias_matches_canonical_method():
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, 1.0, ALIAS_VF[0])
    nx.set_node_attributes(graph, 0.0, ALIAS_THETA[0])
    partitioner = FractalPartitioner(use_spatial_index=False)

    affinity = partitioner._compute_community_affinity(graph, {0}, 1)
    assert affinity == pytest.approx(1.0)
    assert partitioner._compute_community_coherence(
        graph, {0}, 1
    ) == pytest.approx(affinity)


def test_partitioner_spatial_index_respects_phase_wrap():
    partitioner = FractalPartitioner()
    if not partitioner.use_spatial_index:
        pytest.skip("SciPy KDTree is unavailable")

    graph = nx.empty_graph(3)
    phases = {0: math.pi - 0.01, 1: -math.pi + 0.01, 2: 0.0}
    nx.set_node_attributes(graph, 1.0, ALIAS_VF[0])
    nx.set_node_attributes(graph, phases, ALIAS_THETA[0])
    partitioner._build_spatial_index(graph)

    index_by_node = {
        node: index for index, node in partitioner._node_index_map.items()
    }
    coords = partitioner._kdtree.data
    wrap_distance = math.dist(
        coords[index_by_node[0]], coords[index_by_node[1]]
    )
    opposite_distance = math.dist(
        coords[index_by_node[0]], coords[index_by_node[2]]
    )
    assert wrap_distance < opposite_distance


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf, True, "0.3"])
def test_partitioner_rejects_invalid_coherence_threshold(threshold):
    with pytest.raises(ValueError, match="finite real in \\[0, 1\\]"):
        FractalPartitioner(coherence_threshold=threshold)


@pytest.mark.parametrize(
    ("aliases", "value", "message"),
    [
        (ALIAS_VF, -0.1, "nu_f.*nonnegative"),
        (ALIAS_VF, math.nan, "nu_f.*finite real"),
        (ALIAS_VF, math.inf, "nu_f.*finite real"),
        (ALIAS_VF, True, "nu_f.*finite real"),
        (ALIAS_THETA, math.nan, "phase.*finite real"),
        (ALIAS_THETA, -math.inf, "phase.*finite real"),
        (ALIAS_THETA, False, "phase.*finite real"),
    ],
)
def test_partitioner_rejects_invalid_canonical_coordinates(
    aliases, value, message
):
    graph = nx.empty_graph(1)
    graph.nodes[0][aliases[0]] = value
    with pytest.raises(ValueError, match=message):
        FractalPartitioner(use_spatial_index=False).partition_network(graph)


def test_partitioner_canonical_zero_precedes_invalid_legacy_aliases():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_VF[0]: 0.0,
                ALIAS_VF[-1]: math.nan,
                ALIAS_THETA[0]: 0.0,
                ALIAS_THETA[-1]: math.inf,
            }
        )
    partitioner = FractalPartitioner(use_spatial_index=False)
    partitions = partitioner.partition_network(graph)
    assert len(partitions) == 1
    assert partitioner._compute_community_coherence(graph, {0}, 1) == pytest.approx(1.0)
