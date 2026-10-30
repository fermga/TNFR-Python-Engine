"""Auxiliary structural-affinity and Hamiltonian boundary regressions."""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction

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


@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("sign", (-1, 1))
def test_affinity_admission_rejects_nonzero_underflow_before_zero_projection(
    kind, sign
):
    tiny = Fraction(sign, 10**1000)
    weights = [tiny, 0.0, 1.0] if kind == "dense" else [(1, 0, tiny), (1, 2, 1.0)]
    with pytest.raises(ValueError, match="underflows"):
        local_phase_sync_weighted(
            _phase_sync_graph(), 1, nodes_order=[0, 1, 2], W_row=weights
        )


@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize(
    "left,right",
    [(1e308, 1e308), (1e308, 8e307), (math.ulp(0.0), 2 * math.ulp(0.0))],
)
def test_extreme_finite_weights_preserve_the_analytic_orthogonal_mean(
    kind, left, right
):
    graph = _phase_sync_graph()
    graph.nodes[2]["phase"] = math.pi / 2
    weights = [left, 0.0, right] if kind == "dense" else [(1, 0, left), (1, 2, right)]
    # Two perpendicular unit vectors: ||a*u+b*v||/(a+b).
    ratio = right / left
    expected = math.sqrt(1 + ratio * ratio) / (1 + ratio)
    actual = local_phase_sync_weighted(graph, 1, nodes_order=[0, 1, 2], W_row=weights)
    assert actual == pytest.approx(expected, abs=2e-16)


def test_subnormal_single_weight_retains_unit_vector_magnitude():
    graph = _phase_sync_graph()
    graph.nodes[2]["phase"] = 1.0
    assert local_phase_sync_weighted(
        graph, 1, nodes_order=[0, 1, 2], W_row=[0.0, 0.0, math.ulp(0.0)]
    ) == pytest.approx(1.0, abs=2e-16)


@pytest.mark.parametrize(
    "weights",
    ([1e308, 0.0, 1e308], [(1, 0, 1e308), (1, 2, 1e308)]),
)
def test_overflowing_aligned_weight_sum_still_has_unit_synchrony(weights):
    assert (
        local_phase_sync_weighted(
            _phase_sync_graph(), 1, nodes_order=[0, 1, 2], W_row=weights
        )
        == 1.0
    )


def test_all_row_reader_keeps_mixed_order_duplicates_and_live_phase_admission():
    from tnfr.metrics.coherence import _all_local_phase_sync_weighted

    graph = nx.Graph()
    order = ["z", 7, "a"]
    graph.add_nodes_from(order, phase=0.0)
    graph.add_edges_from([("z", 7), (7, "a")])
    payload = [(1, 2, 1.0), (1, 1, 7.0), (1, 0, 1.0), (1, 2, 2.0)]
    initial = _all_local_phase_sync_weighted(graph, order, payload)
    assert list(initial) == order
    assert initial == {"z": 0.0, 7: 1.0, "a": 0.0}
    graph.nodes["a"]["phase"] = math.pi / 2
    changed = _all_local_phase_sync_weighted(graph, order, payload)
    assert changed[7] == pytest.approx(math.sqrt(10) / 4)
    assert changed[7] == local_phase_sync_weighted(
        graph, 7, nodes_order=order, W_row=payload
    )
    graph.nodes["a"]["phase"] = math.nan
    with pytest.raises(ValueError, match="phase"):
        _all_local_phase_sync_weighted(graph, order, payload)
    with pytest.raises(ValueError, match="phase"):
        local_phase_sync_weighted(graph, 7, nodes_order=order, W_row=payload)


def test_all_row_reader_admits_invalid_weights_outside_the_first_row():
    from tnfr.metrics.coherence import _all_local_phase_sync_weighted

    with pytest.raises(ValueError, match="underflows"):
        _all_local_phase_sync_weighted(
            _phase_sync_graph(),
            [0, 1, 2],
            [(0, 1, 1.0), (2, 0, Fraction(-1, 10**1000))],
        )


def test_warm_pulse_observation_reads_direct_support_and_node_changes():
    from tnfr.physics.structural_diffusion import compute_nodal_pulse

    graph = nx.empty_graph(3)
    attrs = dict(EPI=0.5, nu_f=1.0, phase=0.0, Si=0.8, delta_nfr=0.0)
    for node in graph:
        graph.nodes[node].update(attrs)
    graph.graph["COHERENCE"] = {
        "enabled": True,
        "scope": "neighbors",
        "store_mode": "sparse",
    }
    assert compute_nodal_pulse(graph)["mean_local_resonance"] == 0.0
    graph.add_edge(0, 1)
    # Two aligned one-neighbor rows and one isolated row.
    assert compute_nodal_pulse(graph)["mean_local_resonance"] == pytest.approx(2 / 3)
    graph.add_node("new", **attrs)
    graph.add_edge(2, "new")
    observed = compute_nodal_pulse(graph)
    assert observed["mean_local_resonance"] == 1.0
    assert observed["n_nodes"] == 4


@pytest.mark.parametrize("consumer", ("pulse", "dissonance"))
def test_all_node_observation_captures_live_phase_and_support_a_bounded_number_of_times(
    monkeypatch, consumer
):
    import tnfr.metrics.trig_cache as trig_module
    import tnfr.utils.cache as cache_module
    from tnfr.physics.structural_diffusion import compute_nodal_pulse

    graph = _configured_path()
    graph.graph["COHERENCE"]["store_mode"] = "sparse"
    graph.add_nodes_from(range(12))
    for node in graph:
        graph.nodes[node].update(EPI=0.5, nu_f=1.0, phase=0.0, Si=0.8, delta_nfr=0.0)
    observe = compute_nodal_pulse if consumer == "pulse" else dissonance_events
    observe(graph)
    counters = {"support": 0, "phase": 0}
    original_support = cache_module._edge_source_snapshot
    original_phase = trig_module._graph_theta_checksums

    def support(source):
        counters["support"] += 1
        return original_support(source)

    def phase(source):
        counters["phase"] += 1
        return original_phase(source)

    monkeypatch.setattr(cache_module, "_edge_source_snapshot", support)
    monkeypatch.setattr(trig_module, "_graph_theta_checksums", phase)
    observe(graph)
    assert 0 < counters["support"] <= 4
    assert 0 < counters["phase"] <= 4
    # A later direct source edit still invalidates the next captured state.
    graph.add_edge(0, 1)
    graph.nodes[0]["phase"] = math.nan
    with pytest.raises(ValueError, match="phase"):
        observe(graph)


def test_sparse_row_partition_preserves_order_and_public_reduction() -> None:
    from tnfr.metrics.coherence import _partition_sparse_phase_sync_weights

    graph = _phase_sync_graph()
    graph.nodes[2]["phase"] = math.pi / 2
    payload = [
        (2, 0, 0.5),
        (1, 2, 1.0),
        (1, 1, 7.0),
        (1, 0, 1.0),
        (1, 2, 2.0),
        (0, 2, 0.0),
    ]
    prior = list(payload)

    rows = _partition_sparse_phase_sync_weights(payload, 3)

    assert rows is not None
    assert rows == [
        [(0, 2, 0.0)],
        [(1, 2, 1.0), (1, 1, 7.0), (1, 0, 1.0), (1, 2, 2.0)],
        [(2, 0, 0.5)],
    ]
    assert payload == prior
    for node in graph:
        assert local_phase_sync_weighted(
            graph, node, nodes_order=[0, 1, 2], W_row=rows[node]
        ) == local_phase_sync_weighted(
            graph, node, nodes_order=[0, 1, 2], W_row=payload
        )
    assert local_phase_sync_weighted(
        graph, 1, nodes_order=[0, 1, 2], W_row=rows[1]
    ) == pytest.approx(math.sqrt(10.0) / 4.0)


@pytest.mark.parametrize(
    "invalid_entry",
    (
        (2, 3, 1.0),
        (-1, 0, 1.0),
        (2, 0, math.nan),
        (2, 0, math.inf),
        (2, 2, -1.0),
    ),
)
def test_sparse_partition_rejects_invalid_entries_outside_first_row(
    invalid_entry,
) -> None:
    from tnfr.metrics.coherence import _partition_sparse_phase_sync_weights

    with pytest.raises(ValueError):
        _partition_sparse_phase_sync_weights([(0, 1, 1.0), invalid_entry], 3)


def test_sparse_partition_does_not_cache_phase_state() -> None:
    from tnfr.metrics.coherence import _partition_sparse_phase_sync_weights

    graph = _phase_sync_graph()
    rows = _partition_sparse_phase_sync_weights([(1, 0, 1.0), (1, 2, 1.0)], 3)
    assert rows is not None
    assert local_phase_sync_weighted(
        graph, 1, nodes_order=[0, 1, 2], W_row=rows[1]
    ) == pytest.approx(1.0)
    graph.nodes[2]["phase"] = math.nan
    with pytest.raises(ValueError, match="phase"):
        local_phase_sync_weighted(graph, 1, nodes_order=[0, 1, 2], W_row=rows[1])


@pytest.mark.parametrize("consumer", ("pulse", "dissonance"))
def test_all_node_observation_admits_sparse_entries_in_linear_work(
    monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    import tnfr.metrics.coherence as affinity_module
    from tnfr.physics.structural_diffusion import compute_nodal_pulse

    graph = nx.complete_graph(8)
    graph.graph["COHERENCE"] = {
        "enabled": True,
        "scope": "neighbors",
        "store_mode": "sparse",
    }
    for node in graph:
        graph.nodes[node].update(EPI=0.5, nu_f=1.0, phase=0.0, Si=0.8, delta_nfr=0.0)
    original = affinity_module._validated_affinity_weight
    admissions = 0

    def counted(value):
        nonlocal admissions
        admissions += 1
        return original(value)

    monkeypatch.setattr(affinity_module, "_validated_affinity_weight", counted)

    if consumer == "pulse":
        assert compute_nodal_pulse(graph)["mean_local_resonance"] == pytest.approx(1.0)
    else:
        dissonance_events(graph)
        assert not graph.graph["history"].get("events")

    entries = 8 * 7
    assert entries <= admissions <= 2 * entries


def test_local_phase_sync_query_does_not_record_affinity_history() -> None:
    from tnfr.metrics.coherence import local_phase_sync

    graph = _phase_sync_graph()
    history = ensure_history(graph)
    before = {key: list(history.get(key, ())) for key in ("W_sparse", "W_i", "W_stats")}

    assert local_phase_sync(graph, 1) == pytest.approx(1.0)
    assert {
        key: list(history.get(key, ())) for key in ("W_sparse", "W_i", "W_stats")
    } == before


def test_dissonance_events_builds_affinity_once_without_history_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _configured_path()
    history = ensure_history(graph)
    before = {key: list(history.get(key, ())) for key in ("W_sparse", "W_i", "W_stats")}
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
        key: list(history.get(key, ())) for key in ("W_sparse", "W_i", "W_stats")
    } == before


@pytest.mark.parametrize(
    ("markers", "context", "expected_step"),
    [
        ({}, None, 2),
        ({"_runtime_step_next": 7}, None, 7),
        ({}, {"step": 9}, 9),
    ],
)
def test_dissonance_event_ordinal_uses_callback_or_runtime_before_legacy_samples(
    markers, context, expected_step
) -> None:
    graph = nx.empty_graph(1)
    graph.nodes[0].update(EPI=0.0, nu_f=1.0, theta=0.0, **{ALIAS_DNFR[0]: 1.0})
    graph.graph.update(HISTORY_MAXLEN=2, history={"C_steps": [0.7, 0.8]}, **markers)

    dissonance_events(graph, context)

    assert list(graph.graph["history"]["events"]) == [
        ("dissonance_start", {"node": 0, "step": expected_step})
    ]
