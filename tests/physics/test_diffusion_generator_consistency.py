"""Independent matrix checks for weighted, directed and heterogeneous diffusion."""

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.structural_diffusion import (
    degree_weighted_total,
    dispersion_relation,
    instability_threshold,
    relaxation_spectrum,
    structural_diffusion_operator,
    structural_eigenmodes,
    symmetric_normalized_laplacian,
    verify_structural_diffusion,
    verify_structural_stability,
)


def _triad(graph, frequencies, epi=None):
    for index, node in enumerate(graph):
        graph.nodes[node].update(
            nu_f=frequencies[index], EPI=float(index) if epi is None else epi[index],
            theta=0.0,
        )
    return graph


def test_heterogeneous_rates_use_actual_nodal_generator():
    graph = _triad(nx.path_graph(3), [1.0, 3.0, 5.0])
    # Characteristic polynomial of diag(1,3,5)*L_rw is x*(x-2)*(x-7).
    assert relaxation_spectrum(graph) == pytest.approx([0.0, 2.0, 7.0], abs=1e-12)
    assert dispersion_relation(graph, 0.5) == pytest.approx([0.5, -1.5, -6.5])
    assert instability_threshold(graph) == pytest.approx(2.0)


def test_frequency_change_invalidates_rates_without_changing_geometry():
    graph = _triad(nx.path_graph(3), [1.0, 1.0, 1.0])
    assert relaxation_spectrum(graph) == pytest.approx([0.0, 1.0, 2.0], abs=1e-12)
    graph.nodes[1]["nu_f"] = 3.0
    assert relaxation_spectrum(graph) == pytest.approx([0.0, 1.0, 4.0], abs=1e-12)


@pytest.mark.parametrize("graph_type", [nx.MultiGraph, nx.MultiDiGraph])
def test_parallel_weighted_laplacian_matches_independent_adjacency(graph_type):
    graph = graph_type()
    graph.add_weighted_edges_from([(0, 0, 0.25), (0, 1, 0.5), (0, 1, 1.5), (0, 2, 3.0)])
    adjacency = nx.to_numpy_array(graph, weight="weight")
    degree = adjacency.sum(axis=1)
    expected = np.eye(len(graph)) - np.divide(
        adjacency, degree[:, None], out=np.zeros_like(adjacency), where=degree[:, None] > 0,
    )
    expected[degree == 0] = 0.0
    _, actual = structural_diffusion_operator(graph)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_directed_relaxation_does_not_use_symmetric_eigensolver():
    graph = _triad(nx.DiGraph([(0, 1), (1, 2), (2, 0)]), [1.0] * 3)
    # Re eigenvalues of I-P for the directed 3-cycle.
    assert relaxation_spectrum(graph) == pytest.approx([0.0, 1.5, 1.5], abs=1e-12)
    with pytest.raises(ValueError, match="symmetric"):
        symmetric_normalized_laplacian(graph)


def test_certificate_uses_frequency_in_sampled_flow():
    graph = _triad(nx.path_graph(2), [2.0, 2.0], [1.0, 0.0])
    certificate = verify_structural_diffusion(graph, dt=0.1, steps=1)
    # x <- x - dt*diag(nu_f)*L*x = [.8,.2], standard deviation .3.
    assert certificate.final_field_std == pytest.approx(0.3)


def test_heterogeneous_conservation_is_not_mislabelled_degree_conservation():
    graph = _triad(nx.path_graph(2), [1.0, 3.0], [1.0, 0.0])
    certificate = verify_structural_diffusion(graph, dt=0.1, steps=100)
    assert not certificate.degree_weighted_conserved
    assert certificate.invariant_weighted_conserved
    assert certificate.is_valid_diffusion
    assert certificate.max_conservation_drift == pytest.approx(0.5)


def test_disconnected_stationary_field_is_valid_without_global_uniformity():
    graph = _triad(nx.Graph([(0, 1), (2, 3)]), [1.0] * 4, [0.0, 0.0, 1.0, 1.0])
    certificate = verify_structural_diffusion(graph)
    assert certificate.dnfr_is_graph_laplacian
    assert not certificate.relaxes_to_uniform
    assert certificate.reaches_stationary_state
    assert certificate.is_valid_diffusion


def test_frozen_frequency_preserves_nonuniform_field():
    graph = _triad(nx.path_graph(2), [0.0, 0.0], [1.0, 0.0])
    certificate = verify_structural_diffusion(graph)
    assert certificate.final_field_std == pytest.approx(0.5)
    assert certificate.reaches_stationary_state


def test_weighted_total_sums_parallel_edges():
    graph = nx.MultiGraph()
    graph.add_weighted_edges_from([(0, 1, 0.5), (0, 1, 1.5)])
    _triad(graph, [1.0, 1.0], [1.0, 3.0])
    assert degree_weighted_total(graph) == pytest.approx(8.0)


@pytest.mark.parametrize("frequency", [-1.0, float("nan"), float("inf")])
def test_invalid_frequency_cannot_receive_relaxation_certificate(frequency):
    graph = _triad(nx.path_graph(2), [1.0, frequency])
    with pytest.raises(ValueError, match="frequency"):
        relaxation_spectrum(graph)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_weight_cannot_receive_diffusion_certificate(weight):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=weight)
    with pytest.raises(ValueError, match="weight"):
        structural_diffusion_operator(graph)


@pytest.mark.parametrize("frequencies", [[1.0, 3.0, 5.0], [0.0, 0.0, 0.0]])
def test_fiedler_certificate_rejects_unsupported_capacity(frequencies):
    graph = _triad(nx.path_graph(3), frequencies)
    with pytest.raises(ValueError, match="common positive frequency"):
        verify_structural_stability(graph)


def test_fiedler_certificate_requires_connected_positive_support():
    graph = _triad(nx.Graph([(0, 1), (2, 3)]), [1.0] * 4)
    with pytest.raises(ValueError, match="connected"):
        verify_structural_stability(graph)


@pytest.mark.parametrize("epi", [float("nan"), float("inf")])
def test_nonfinite_field_cannot_receive_diffusion_certificate(epi):
    graph = _triad(nx.path_graph(2), [1.0, 1.0], [0.0, epi])
    with pytest.raises(ValueError, match="Non-finite"):
        verify_structural_diffusion(graph)


def test_copied_undirected_cache_cannot_certify_asymmetric_geometry():
    graph = nx.path_graph(3)
    structural_eigenmodes(graph)
    directed = graph.to_directed()
    directed.remove_edges_from([(1, 0), (2, 1)])
    with pytest.raises(ValueError, match="symmetric"):
        structural_eigenmodes(directed)


def test_caller_cannot_corrupt_cached_geometry_modes():
    graph = nx.path_graph(3)
    eigenvalues, eigenvectors = structural_eigenmodes(graph)
    eigenvalues[:] = 99.0
    eigenvectors[:] = 0.0
    fresh_values, fresh_vectors = structural_eigenmodes(graph)
    assert fresh_values == pytest.approx([0.0, 1.0, 2.0], abs=1e-12)
    np.testing.assert_allclose(fresh_vectors.T @ fresh_vectors, np.eye(3), atol=1e-12)
