"""Auxiliary structural-affinity and Hamiltonian boundary regressions."""

from __future__ import annotations

from copy import deepcopy
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR
from tnfr.glyph_history import ensure_history
from tnfr.metrics.coherence import coherence_matrix, local_phase_sync_weighted
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.diagnosis import dissonance_events
from tnfr.operators.hamiltonian import (
    InternalHamiltonian,
    build_H_coherence,
    build_H_coupling,
)


def _configured_path() -> nx.Graph:
    graph = nx.path_graph(("left", "centre", "right"))
    graph.graph["COHERENCE"] = {
        "enabled": True,
        "scope": "neighbors",
        "store_mode": "dense",
        "self_on_diag": True,
    }
    for index, node in enumerate(graph):
        graph.nodes[node].update(
            EPI=0.5,
            nu_f=1.0,
            phase=0.0,
            Si=0.8,
            **{
                ALIAS_DNFR[0]: float(index),
                ALIAS_DEPI[0]: 0.0,
            },
        )
    return graph


def test_affinity_is_not_a_psd_or_trace_definition_of_canonical_coherence() -> None:
    graph = _configured_path()

    nodes, payload = coherence_matrix(graph, use_numpy=True)
    matrix = np.asarray(payload, dtype=float)

    assert nodes == ["left", "centre", "right"]
    assert matrix.tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    assert np.linalg.eigvalsh(matrix)[0] == pytest.approx(1.0 - np.sqrt(2.0))
    assert np.trace(matrix) / len(matrix) == 1.0
    assert compute_coherence(graph) == pytest.approx(0.5)


def test_hamiltonian_builder_handles_dense_three_by_three_affinity() -> None:
    graph = _configured_path()

    result = build_H_coherence(graph, C_0=-0.25)

    assert result.shape == (3, 3)
    assert np.allclose(result, result.conj().T)
    assert result[0, 1] == pytest.approx(-0.25)


def test_hamiltonian_builder_handles_python_dense_three_by_three_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _configured_path()
    import tnfr.metrics.coherence as affinity_module

    original = affinity_module.coherence_matrix
    monkeypatch.setattr(
        affinity_module,
        "coherence_matrix",
        lambda source, **kwargs: original(
            source,
            use_numpy=False,
            **kwargs,
        ),
    )

    result = build_H_coherence(graph, C_0=-0.25)

    assert result.shape == (3, 3)
    assert np.allclose(result, result.conj().T)
    assert result[0, 1] == pytest.approx(-0.25)


def test_sparse_default_is_a_zero_free_off_diagonal_projection() -> None:
    dense_graph = _configured_path()
    sparse_graph = deepcopy(dense_graph)
    sparse_graph.graph["COHERENCE"]["store_mode"] = "sparse"

    dense_nodes, dense_payload = coherence_matrix(dense_graph, use_numpy=True)
    sparse_nodes, sparse_payload = coherence_matrix(sparse_graph, use_numpy=True)
    restored = np.zeros((3, 3), dtype=float)
    for row, column, value in sparse_payload:
        restored[row, column] = value

    assert sparse_nodes == dense_nodes
    dense = np.asarray(dense_payload, dtype=float)
    np.fill_diagonal(dense, 0.0)
    assert np.array_equal(restored, dense)
    assert len(sparse_payload) == 4
    assert all(value != 0.0 for _, _, value in sparse_payload)
    assert all(row != column for row, column, _ in sparse_payload)


def test_hamiltonian_uses_full_affinity_not_sparse_storage_threshold() -> None:
    graph = _configured_path()
    graph.nodes["right"]["phase"] = math.pi
    graph.graph["COHERENCE"].update(
        store_mode="sparse",
        threshold=0.8,
    )

    _, sparse_payload = coherence_matrix(graph, use_numpy=True)
    sparse_entries = {(row, column) for row, column, _ in sparse_payload}
    hamiltonian = build_H_coherence(graph, C_0=1.0)

    assert (1, 2) not in sparse_entries
    assert hamiltonian[1, 2] == pytest.approx(0.75)


def test_hamiltonian_builder_reorders_affinity_to_requested_nodes() -> None:
    graph = _configured_path()
    canonical_nodes = list(graph)
    reordered_nodes = ["right", "left", "centre"]
    permutation = [canonical_nodes.index(node) for node in reordered_nodes]

    canonical = build_H_coherence(graph, nodes=canonical_nodes, C_0=1.0)
    reordered = build_H_coherence(graph, nodes=reordered_nodes, C_0=1.0)

    assert np.array_equal(
        reordered,
        canonical[np.ix_(permutation, permutation)],
    )


def test_disabled_affinity_contributes_a_zero_hamiltonian() -> None:
    graph = _configured_path()
    graph.graph["COHERENCE"] = {"enabled": False}

    standalone = build_H_coherence(graph)
    internal = InternalHamiltonian(graph)

    assert np.array_equal(standalone, np.zeros((3, 3), dtype=complex))
    assert np.array_equal(internal.H_coh, standalone)


@pytest.mark.parametrize("nodes", [["left", "centre"], ["left"] * 3])
def test_hamiltonian_builder_rejects_incomplete_or_duplicate_node_orders(
    nodes,
) -> None:
    with pytest.raises(ValueError, match="duplicate-free permutation"):
        build_H_coherence(_configured_path(), nodes=nodes)


@pytest.mark.parametrize("strength", [True, float("nan"), float("inf"), "-1"])
def test_hamiltonian_builder_rejects_invalid_strength(strength) -> None:
    with pytest.raises((TypeError, ValueError), match="C_0"):
        build_H_coherence(_configured_path(), C_0=strength)


_GRAPH_TYPES = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


@pytest.mark.parametrize("graph_type", _GRAPH_TYPES)
def test_coupling_uses_underlying_simple_undirected_support(graph_type) -> None:
    graph = graph_type()
    graph.add_nodes_from(("left", "right"))
    graph.add_edge("left", "right")
    if graph.is_directed():
        graph.add_edge("right", "left")
    if graph.is_multigraph():
        graph.add_edge("left", "right")
    graph.add_edge("left", "left")

    result = build_H_coupling(
        graph,
        nodes=["right", "left"],
        J_0=0.25,
    )

    assert np.array_equal(
        result,
        np.asarray([[0.0, 0.25], [0.25, 0.25]], dtype=complex),
    )


@pytest.mark.parametrize("graph_type", _GRAPH_TYPES)
def test_internal_and_standalone_coupling_builders_agree(graph_type) -> None:
    graph = graph_type()
    graph.add_nodes_from(("left", "right"))
    graph.add_edge("left", "right")
    graph.add_edge("left", "left")
    graph.graph["COHERENCE"] = {"enabled": False}
    graph.graph["H_COUPLING_STRENGTH"] = -0.375

    internal = InternalHamiltonian(graph)
    standalone = build_H_coupling(
        graph,
        nodes=list(internal.nodes),
        J_0=-0.375,
    )

    assert np.array_equal(internal.H_coupling, standalone)


@pytest.mark.parametrize(
    "strength",
    [True, float("nan"), float("inf"), float("-inf"), "0.1", 0.1j],
)
def test_coupling_builder_rejects_invalid_strength(strength) -> None:
    graph = nx.Graph()
    graph.add_edge("left", "right")

    with pytest.raises((TypeError, ValueError), match="J_0"):
        build_H_coupling(graph, J_0=strength)


@pytest.mark.parametrize(
    "nodes",
    [["left"], ["left", "left"], ["left", "right", "extra"]],
)
def test_coupling_builder_rejects_invalid_node_orders(nodes) -> None:
    graph = nx.Graph()
    graph.add_edge("left", "right")

    with pytest.raises(ValueError, match="duplicate-free permutation"):
        build_H_coupling(graph, nodes=nodes)


def _phase_sync_graph() -> nx.Graph:
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node]["phase"] = 0.0
    return graph


@pytest.mark.parametrize(
    "weights",
    [
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
        [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
        [(1, 0, 1.0), (1, 2, 1.0)],
    ],
)
def test_weighted_phase_sync_accepts_dense_and_sparse_payloads(weights) -> None:
    result = local_phase_sync_weighted(
        _phase_sync_graph(),
        1,
        nodes_order=[0, 1, 2],
        W_row=weights,
    )

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.asarray([1.0, 0.0]), "wrong length"),
        (np.ones((2, 2)), "wrong shape"),
        (np.ones((3, 3, 1)), "one or two dimensions"),
        ([1.0, float("nan"), 1.0], "finite and nonnegative"),
        ([1.0, -0.1, 1.0], "finite and nonnegative"),
        ([(1, 3, 1.0)], "index is out of range"),
    ],
)
def test_weighted_phase_sync_rejects_invalid_payloads(weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        local_phase_sync_weighted(
            _phase_sync_graph(),
            1,
            nodes_order=[0, 1, 2],
            W_row=weights,
        )


def test_local_phase_sync_query_does_not_record_affinity_history() -> None:
    from tnfr.metrics.coherence import local_phase_sync

    graph = _phase_sync_graph()
    history = ensure_history(graph)
    before = {
        key: list(history.get(key, ()))
        for key in ("W_sparse", "W_i", "W_stats")
    }

    assert local_phase_sync(graph, 1) == pytest.approx(1.0)
    assert {
        key: list(history.get(key, ()))
        for key in ("W_sparse", "W_i", "W_stats")
    } == before


def test_dissonance_events_builds_affinity_once_without_history_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _configured_path()
    history = ensure_history(graph)
    before = {
        key: list(history.get(key, ()))
        for key in ("W_sparse", "W_i", "W_stats")
    }
    import tnfr.metrics.diagnosis as diagnosis_module

    original = diagnosis_module.coherence_matrix
    calls = 0

    def counted(source, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["_record_history"] is False
        return original(source, **kwargs)

    monkeypatch.setattr(diagnosis_module, "coherence_matrix", counted)

    dissonance_events(graph)

    assert calls == 1
    assert {
        key: list(history.get(key, ()))
        for key in ("W_sparse", "W_i", "W_stats")
    } == before