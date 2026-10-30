"""Raw transport scalars cannot acquire validity through aggregation or caches."""

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics._conductance import read_conductance
from tnfr.physics.canonical import estimate_coherence_length_with_provenance
from tnfr.physics.structural_diffusion import (
    current_divergence,
    structural_diffusion_operator,
    structural_eigenvalues,
)


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), "1", math.nan, math.inf, Fraction(1, 2**2000)],
)
def test_raw_weights_reject_before_parallel_aggregation_or_materialization(kind, bad):
    graph = kind()
    graph.add_edge(0, 1, weight=bad, length=1.0)
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=1.0, length=1.0)
    with pytest.raises(ValueError, match="finite nonnegative edge weights"):
        read_conductance(graph)


@pytest.mark.parametrize("reader", [current_divergence, structural_diffusion_operator])
@pytest.mark.parametrize("bad", [True, "1", Fraction(1, 2**2000)])
def test_shared_diffusion_readouts_cannot_consume_coerced_weights(reader, bad):
    graph = nx.path_graph(2)
    graph[0][1].update(weight=bad, length=1.0)
    nx.set_node_attributes(graph, {0: 1.0, 1: 0.0}, "EPI")
    with pytest.raises(ValueError, match="finite nonnegative edge weights"):
        reader(graph)


@pytest.mark.parametrize("kind", [nx.Graph, nx.MultiGraph])
@pytest.mark.parametrize("bad", [True, "1", Fraction(1, 2**2000)])
def test_invalid_mutation_cannot_reuse_a_valid_spectrum_or_fallback(kind, bad):
    graph = kind()
    graph.add_edge(0, 1, weight=1.0, length=1.0)
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=1.0, length=1.0)
    initial = estimate_coherence_length_with_provenance(graph)
    assert initial.method == "spectral_gap"
    cached = graph.graph["_tnfr_spectrum_cache"]
    data = graph[0][1][0] if graph.is_multigraph() else graph[0][1]
    data["weight"] = bad
    with pytest.raises(ValueError, match="finite nonnegative edge weights"):
        structural_eigenvalues(graph)
    unavailable = estimate_coherence_length_with_provenance(graph)
    assert unavailable.method == "unavailable"
    assert math.isnan(unavailable.value)
    assert graph.graph["_tnfr_spectrum_cache"] is cached
    data["weight"] = 1.0
    assert estimate_coherence_length_with_provenance(graph) == initial


def test_symmetric_effective_directed_conductance_supports_the_fallback():
    graph = nx.DiGraph([(0, 1), (1, 0)])
    estimate = estimate_coherence_length_with_provenance(graph)
    assert estimate.method == "spectral_gap"
    assert estimate.value == pytest.approx(1 / math.sqrt(2))
    graph.remove_edge(1, 0)
    estimate = estimate_coherence_length_with_provenance(graph)
    assert estimate.method == "unavailable"
    assert math.isnan(estimate.value)


def test_signed_parallel_numpy_integers_preserve_exact_aggregation():
    graph = nx.MultiDiGraph()
    # Summing in np.int64 would overflow at the second addition. The actual
    # effective conductance is exactly 2, matching the reciprocal arc.
    largest = np.iinfo(np.int64).max
    for value in (np.int64(largest), np.int64(2), np.int64(-largest)):
        graph.add_edge(0, 1, weight=value, length=1.0)
    graph.add_edge(1, 0, weight=2, length=1.0)
    np.testing.assert_array_equal(
        read_conductance(graph, symmetric=True).dense(), [[0, 2], [2, 0]]
    )
    assert estimate_coherence_length_with_provenance(graph).value == pytest.approx(
        1 / math.sqrt(2)
    )


def test_exact_parallel_residual_cannot_underflow_into_absent_transport():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, weight=Fraction(2**2000 + 1, 2**2000))
    graph.add_edge(0, 1, weight=-1)
    with pytest.raises(ValueError, match="underflows"):
        read_conductance(graph)


def test_induced_adjacency_admits_only_consumed_edges():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=Fraction(1, 2))
    graph.add_edge(1, 2, weight=True)
    np.testing.assert_array_equal(
        read_conductance(graph, [1, 0]).dense(), [[0, 0.5], [0.5, 0]]
    )
