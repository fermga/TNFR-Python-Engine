"""Independent graph-metric and numerical controls for potential read-outs."""

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.config import get_precision_mode, set_precision_mode
from tnfr.physics import canonical
from tnfr.physics.fields import classify_nodal_topology
from tnfr.physics.vectorized_ops import compute_phi_s_exact_vectorized
from tnfr.utils.cache import reset_global_cache


@pytest.fixture(autouse=True)
def reset_fields():
    previous = get_precision_mode()
    reset_global_cache()
    yield
    reset_global_cache()
    set_precision_mode(previous)


def _pair(length, pressure, attribute="weight"):
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, length, attribute)
    nx.set_node_attributes(graph, pressure, "delta_nfr")
    return graph


@pytest.mark.parametrize("attribute", ["length", "weight"])
@pytest.mark.parametrize(
    "bad", [-1.0, True, np.bool_(True), math.inf, math.nan, "1", Fraction(1, 10**500)]
)
@pytest.mark.parametrize(
    "reader",
    [
        canonical.compute_structural_potential,
        classify_nodal_topology,
        canonical.estimate_coherence_length_with_provenance,
    ],
)
def test_all_geometry_readers_reject_invalid_authoritative_metric(
    attribute, bad, reader
):
    graph = _pair(1.0, 1.0, attribute)
    reader(graph)
    graph.edges[0, 1][attribute] = bad
    with pytest.raises(ValueError, match="edge length"):
        reader(graph)


@pytest.mark.parametrize("attribute", ["length", "weight"])
@pytest.mark.parametrize("mode", ["standard", "high", "research"])
@pytest.mark.parametrize("length,pressure", [(1e200, 1e300), (1e-200, 1e-300)])
@pytest.mark.parametrize("vectorized", [False, True])
def test_representable_inverse_square_product_survives_intermediate_range(
    attribute, mode, length, pressure, vectorized, monkeypatch
):
    set_precision_mode(mode)
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", vectorized)
    graph = _pair(length, pressure, attribute)
    expected = float(Fraction.from_float(pressure) / Fraction.from_float(length) ** 2)
    assert math.isfinite(expected) and expected > 0.0
    assert canonical.compute_structural_potential(graph) == {0: expected, 1: expected}
    assert canonical.compute_structural_potential(graph, landmark_ratio=0.5) == {
        0: expected,
        1: expected,
    }


@pytest.mark.parametrize("vectorized", [False, True])
def test_inverse_square_row_cancellation_occurs_before_overflow_projection(
    vectorized, monkeypatch
):
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", vectorized)
    graph = nx.star_graph(3)
    nx.set_edge_attributes(graph, 1e-155, "weight")
    pressure = {0: 0.0, 1: 1e308, 2: -1e308, 3: 1e-310}
    nx.set_node_attributes(graph, pressure, "delta_nfr")
    expected = float(Fraction.from_float(1e-310) / Fraction.from_float(1e-155) ** 2)
    assert canonical.compute_structural_potential(graph)[0] == expected


def test_subnormal_contributions_are_combined_before_final_rounding():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    graph = nx.star_graph(4)
    nx.set_edge_attributes(graph, 2.0, "length")
    nx.set_node_attributes(graph, tiny, "delta_nfr")
    assert canonical.compute_structural_potential(graph)[0] == tiny


def test_unrepresentably_small_centrality_is_not_reported_as_absent_metric_pairs():
    graph = _pair(1e200, 0.0)
    result = classify_nodal_topology(graph)
    assert result["available"] is False
    assert result["status"] == "centrality_below_represented_range"


def test_zero_source_does_not_need_a_representable_distance_power():
    assert canonical.compute_structural_potential(_pair(1e200, 0.0), alpha=3.0) == {
        0: 0.0,
        1: 0.0,
    }


@pytest.mark.parametrize("attribute", ["length", "weight"])
@pytest.mark.parametrize("vectorized", [False, True])
def test_potential_rejects_reachable_path_overflow(attribute, vectorized, monkeypatch):
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", vectorized)
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1e308, attribute)
    with pytest.raises(ValueError, match="path distance exceeds"):
        canonical.compute_structural_potential(graph)


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_potential_and_topology_retain_outgoing_minimum_parallel_metric_under_relabeling(
    kind,
):
    graph = kind()
    graph.add_edge("source", "middle", length=0.5, weight=0.0)
    graph.add_edge("middle", "sink", length=1.5, weight=10.0)
    graph.add_node("isolated")
    if graph.is_multigraph():
        graph.add_edge("source", "middle", length=7.0, weight=100.0)
    nx.set_node_attributes(graph, 1.0, "delta_nfr")
    expected = {"source": 4.25, "middle": 1 / 1.5**2, "sink": 0.0, "isolated": 0.0}
    if not graph.is_directed():
        expected["middle"] += 4.0
        expected["sink"] = 0.25 + 1 / 1.5**2
    result = canonical.compute_structural_potential(graph)
    assert result == pytest.approx(expected)
    assert classify_nodal_topology(graph)["centrality"] == pytest.approx(expected)
    labels = {
        "source": 7,
        "middle": ("tuple", 1),
        "sink": -3,
        "isolated": frozenset({2}),
    }
    observed = canonical.compute_structural_potential(nx.relabel_nodes(graph, labels))
    assert {node: observed[labels[node]] for node in graph} == result


@pytest.mark.parametrize("bad", [True, np.bool_(True), math.nan, math.inf, "2"])
def test_potential_exponent_requires_an_actual_finite_real(bad):
    with pytest.raises(ValueError, match="alpha"):
        canonical.compute_structural_potential(_pair(1.0, 1.0), alpha=bad)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((1, 1)),
        np.array([[0, -1], [-1, 0]]),
        np.array([[0, 1], [2, 0]]),
        np.array([[0, 1], [1, 0]], dtype=bool),
    ],
)
def test_declared_distance_matrix_uses_the_same_domain_as_coherence_fit(bad):
    graph = _pair(1.0, 1.0)
    with pytest.raises(ValueError, match="distance_matrix"):
        compute_phi_s_exact_vectorized(
            graph, list(graph), {0: 1.0, 1: 1.0}, 2.0, distance_matrix=bad
        )
