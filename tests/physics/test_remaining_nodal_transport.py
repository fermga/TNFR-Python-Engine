"""Scaled transport domains and value-only spectral read-out regressions."""

import copy
from decimal import Decimal, localcontext

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics._weight_normalization import normalize_weights
from tnfr.physics._conductance import ConductanceSnapshot, read_conductance
from tnfr.physics.directed_diffusion import directed_rw_laplacian
from tnfr.physics import structural_diffusion as diffusion


def _star(kind=nx.Graph, scale=1e308):
    graph = kind()
    graph.add_nodes_from(["center", "left", "right", "isolated"])
    graph.add_weighted_edges_from([("center", "left", scale),
                                   ("center", "right", scale)])
    if graph.is_directed():
        graph.add_weighted_edges_from([("left", "center", scale),
                                       ("right", "center", scale)])
    for node, epi, capacity in zip(graph, [0.0, 0.25, -0.25, 2.0], [1., 2., 0., 1.]):
        graph.nodes[node].update(EPI=epi, nu_f=capacity, theta=0.)
    return graph


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("scale", [np.nextafter(0., 1.), 1e-310, 1., 1e308])
def test_graph_and_matrix_normalized_operators_share_the_scaled_domain(kind, scale):
    graph = _star(kind, scale)
    before = copy.deepcopy(graph)
    expected = np.array([[1., -.5, -.5, 0.], [-1., 1., 0., 0.],
                         [-1., 0., 1., 0.], [0., 0., 0., 0.]])
    with np.errstate(all="raise"):
        _, lap = diffusion.structural_diffusion_operator(graph)
        _, transition = diffusion.random_walk_matrix(graph)
        _, symmetric = diffusion.symmetric_normalized_laplacian(graph)
        _, stationary = diffusion.stationary_distribution(graph)
        raw = directed_rw_laplacian(nx.to_numpy_array(graph))
    np.testing.assert_array_equal(lap, expected)
    np.testing.assert_array_equal(lap, raw)
    np.testing.assert_array_equal(transition, np.eye(4) - expected)
    np.testing.assert_array_equal(stationary, [.5, .25, .25, 0.])
    np.testing.assert_array_equal(stationary @ transition, stationary)
    np.testing.assert_array_equal(symmetric, symmetric.T)
    np.testing.assert_allclose(np.linalg.eigvalsh(symmetric), [0., 0., 1., 2.], atol=1e-15)
    assert nx.utils.graphs_equal(graph, before)


def test_overflowing_strength_still_allows_representable_dirichlet_balance():
    graph = _star()
    with np.errstate(all="raise"):
        balance = diffusion.compute_diffusion_energy(graph)
        divergence = diffusion.current_divergence(graph)[1]
        current = diffusion.structural_current(graph)[1]
    np.testing.assert_array_equal(balance.gradient, [0., 2.5e307, -2.5e307, 0.])
    np.testing.assert_array_equal(divergence, balance.gradient)
    np.testing.assert_array_equal(current.sum(axis=1), balance.gradient)
    np.testing.assert_allclose(balance.mobility, [5e-309, 2e-308, 0., 0.], atol=0., rtol=1e-15)
    np.testing.assert_allclose(balance.epi_rate, [0., -.5, 0., 0.], atol=1e-15)
    assert balance.energy == pytest.approx(6.25e306)
    assert balance.energy_rate == pytest.approx(-1.25e307)
    assert balance.energy_rate == float(balance.gradient @ balance.epi_rate)


def test_raw_strength_and_total_do_not_masquerade_as_scaled_outputs():
    graph = _star()
    with pytest.raises(ValueError, match="row strength"):
        read_conductance(graph).strength
    assert diffusion.degree_weighted_total(graph) == 0.
    graph = nx.path_graph(2)
    graph[0][1]["weight"] = 1e308
    nx.set_node_attributes(graph, 1., "EPI")
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="weighted total"):
            diffusion.degree_weighted_total(graph)
        assert diffusion.commute_time(graph)[1][0, 1] == pytest.approx(2.)


def test_weighted_total_retains_cancellation_of_unrepresentable_edge_products():
    graph = nx.path_graph(2)
    graph[0][1]["weight"] = 1e308
    nx.set_node_attributes(graph, {0: 2., 1: -2.}, "EPI")
    with np.errstate(all="raise"):
        assert diffusion.degree_weighted_total(graph) == 0.
    graph = nx.DiGraph()
    for node, value in enumerate([1e308, 1., -1e308]):
        graph.add_edge(node, node, weight=1.)
        graph.nodes[node]["EPI"] = value
    assert diffusion.degree_weighted_total(graph) == 1.


def test_nonzero_weighted_total_below_float_range_is_not_reported_as_zero():
    graph = nx.Graph()
    graph.add_edge(0, 0, weight=1e-308)
    graph.nodes[0]["EPI"] = 1e-308
    with pytest.raises(ValueError, match="weighted total is below"):
        diffusion.degree_weighted_total(graph)


def test_finite_rounded_products_cannot_reverse_the_weighted_total_sign():
    from fractions import Fraction

    graph = nx.Graph()
    values = [float(2**53 + 2), -13510798882111492., .5]
    weights = [1.5, 1., 1.]
    for node, (value, weight) in enumerate(zip(values, weights)):
        graph.add_edge(node, node, weight=weight)
        graph.nodes[node]["EPI"] = value
    exact = sum(Fraction.from_float(weight) * Fraction.from_float(value)
                for weight, value in zip(weights, values))
    assert float(exact) == -.5
    assert diffusion.degree_weighted_total(graph) == float(exact)


@pytest.mark.parametrize("scale", [np.nextafter(0., 1.), 1e-310, 1., 1e308])
@pytest.mark.parametrize("loop", [False, True])
def test_commute_time_uses_cancelled_common_scale_before_raw_resistance(scale, loop):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=scale)
    if loop:
        graph.add_edge(0, 0, weight=scale)
    with np.errstate(all="raise"):
        commute = diffusion.commute_time(graph)[1]
    np.testing.assert_allclose(commute, [[0., 3. if loop else 2.],
                                       [3. if loop else 2., 0.]], atol=1e-14)
    if scale < 1e-308:
        with pytest.raises(ValueError, match="read-out exceeds"):
            diffusion.effective_resistance(graph)
    else:
        with np.errstate(all="raise"):
            resistance = diffusion.effective_resistance(graph)[1]
        assert resistance[0, 1] == pytest.approx(1. / scale, abs=0., rel=1e-14)


def test_scaled_resistance_handles_overflowing_star_degrees_and_disconnected_volume():
    graph = _star()
    with np.errstate(all="raise"):
        resistance = diffusion.effective_resistance(graph)[1]
        commute = diffusion.commute_time(graph)[1]
    np.testing.assert_allclose(resistance[:3, :3],
                               np.array([[0., 1., 1.], [1., 0., 2.], [1., 2., 0.]]) * 1e-308,
                               atol=0., rtol=1e-14)
    np.testing.assert_allclose(commute[:3, :3], [[0., 4., 4.], [4., 0., 8.], [4., 8., 0.]], atol=1e-14)
    assert np.all(np.isinf(commute[:3, 3]))
    assert commute[3, 3] == 0.


def test_unrepresentable_requested_commute_time_remains_an_explicit_error():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1e-308)
    graph.add_edge(0, 0, weight=1e308)
    # The loop is irrelevant to resistance, but its holding time is ~1e616.
    assert diffusion.effective_resistance(graph)[1][0, 1] == pytest.approx(1e308)
    with pytest.raises(ValueError, match="read-out exceeds"):
        diffusion.commute_time(graph)


def test_raw_gradient_overflow_remains_rejected_when_the_walk_is_valid():
    graph = _star()
    nx.set_node_attributes(graph, {"center": 1., "left": 0., "right": 0.}, "EPI")
    assert np.all(np.isfinite(diffusion.random_walk_matrix(graph)[1]))
    with pytest.raises(ValueError, match="divergence"):
        diffusion.current_divergence(graph)
    with pytest.raises(ValueError, match="divergence"):
        diffusion.compute_diffusion_energy(graph)


def test_symmetric_coefficients_are_not_lost_by_squaring_tiny_probabilities():
    tiny = np.nextafter(0., 1.)
    graph = nx.Graph()
    graph.add_edge(0, 0, weight=1e308)
    graph.add_edge(0, 1, weight=tiny)
    with np.errstate(all="raise"):
        _, symmetric = diffusion.symmetric_normalized_laplacian(graph)
        _, transition = diffusion.random_walk_matrix(graph)
    with localcontext() as context:
        context.prec = 80
        expected = float((Decimal.from_float(tiny) / Decimal.from_float(1e308)).sqrt())
    assert transition[0, 1] == 0.  # The probability itself is unrepresentable.
    assert symmetric[0, 1] == symmetric[1, 0] == -expected
    assert symmetric[0, 1] < 0.


def test_loops_and_absorbing_rows_preserve_the_normalized_walk():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 0, weight=1e308)
    graph.add_edge(0, 1, weight=5e307)
    graph.add_edge(0, 1, weight=5e307)
    graph.add_edge(1, 2, weight=0.)
    _, transition = diffusion.random_walk_matrix(graph)
    np.testing.assert_array_equal(transition, [[.5, .5, 0.], [0., 1., 0.], [0., 0., 1.]])
    _, lap = diffusion.structural_diffusion_operator(graph)
    np.testing.assert_array_equal(-np.array([2., 7., 3.]) * (lap @ [0., 1., 0.]), [1., 0., 0.])
    with pytest.raises(ValueError, match="symmetric"):
        diffusion.symmetric_normalized_laplacian(graph)


def test_parallel_aggregate_must_itself_be_finite():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, weight=1e308)
    graph.add_edge(0, 1, weight=1e308)
    with pytest.raises(ValueError, match="finite nonnegative"):
        diffusion.random_walk_matrix(graph)


def test_scaled_stationary_and_energy_vectors_do_not_allocate_dense_matrices(monkeypatch):
    graph = _star()
    monkeypatch.setattr(ConductanceSnapshot, "dense", lambda *args: pytest.fail("dense allocation"))
    assert diffusion.stationary_distribution(graph)[1].sum() == 1.
    assert diffusion.compute_diffusion_energy(graph).energy == pytest.approx(6.25e306)


def test_normalization_is_shared_across_dense_sparse_and_single_row_inputs():
    weights = np.array([[1e308, 1e308, 0.], [0., 0., 0.], [0., 1e-310, 3e-310]])
    before = weights.copy()
    source, target = np.nonzero(weights)
    with np.errstate(all="raise"):
        dense, scale, total = normalize_weights(weights)
        sparse, sparse_scale, sparse_total = normalize_weights(
            weights[source, target], source=source, node_count=3,
        )
        single, single_scale, single_total = normalize_weights(weights[0])
    np.testing.assert_array_equal(sparse, dense[source, target])
    np.testing.assert_array_equal(sparse_scale, scale)
    np.testing.assert_array_equal(sparse_total, total)
    np.testing.assert_array_equal(single, dense[0])
    np.testing.assert_array_equal(single_scale, scale[:1])
    np.testing.assert_array_equal(single_total, total[:1])
    np.testing.assert_allclose(dense, [[.5, .5, 0.], [0., 0., 0.], [0., .25, .75]])
    np.testing.assert_array_equal(weights, before)


@pytest.mark.parametrize("weights", [[-1., 2.], [float("nan")], [float("inf")], [1j]])
def test_shared_normalization_keeps_invalid_conductance_explicit(weights):
    with pytest.raises(ValueError, match="finite nonnegative"):
        normalize_weights(weights)


@pytest.mark.parametrize("graph", [nx.Graph(), nx.empty_graph(1), nx.empty_graph(4),
                                  nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))])
def test_value_only_spectrum_has_no_eigenvector_allocation(graph, monkeypatch):
    nx.set_node_attributes(graph, 2., "nu_f")
    expected = np.linalg.eigvalsh(diffusion.symmetric_normalized_laplacian(graph)[1])
    monkeypatch.setattr(np.linalg, "eigh", lambda *args: pytest.fail("unneeded eigenvectors"))
    rates = diffusion.relaxation_spectrum(graph)
    pulse = diffusion.compute_emergent_pulse(graph)
    np.testing.assert_allclose(rates, 2. * np.maximum(expected, 0.), atol=1e-14)
    assert "vecs" not in graph.graph["_tnfr_spectrum_cache"]
    assert pulse["vibration_energy"] == pytest.approx(.5 * np.maximum(expected, 0.).sum())
    if not graph:
        assert pulse["n_modes"] == pulse["spectral_multiplicity"] == 0


def test_spectrum_cache_upgrades_and_reuses_full_modes_without_aliasing(monkeypatch):
    graph = nx.path_graph(5)
    nx.set_node_attributes(graph, 1., "nu_f")
    values = diffusion.relaxation_spectrum(graph)
    eig, vectors = diffusion.structural_eigenmodes(graph)
    np.testing.assert_allclose(eig, values, atol=1e-14)
    assert graph.graph["_tnfr_spectrum_cache"]["vecs"].shape == (5, 5)
    eig[:] = vectors[:] = 0.
    monkeypatch.setattr(np.linalg, "eigh", lambda *args: pytest.fail("cached full modes"))
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda *args: pytest.fail("cached values"))
    nx.set_node_attributes(graph, 3., "nu_f")
    np.testing.assert_allclose(diffusion.relaxation_spectrum(graph), 3. * values, atol=1e-14)
    assert np.any(diffusion.structural_eigenmodes(graph)[1])
    assert diffusion.compute_emergent_pulse(graph)["n_modes"] == 4


def test_value_cache_invalidates_for_direct_weight_changes_and_parallel_exact_sum():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key="large", weight=2**53 + 1)
    graph.add_edge(0, 1, key="cancel", weight=-2**53)
    nx.set_node_attributes(graph, 1., "nu_f")
    np.testing.assert_allclose(diffusion.relaxation_spectrum(graph), [0., 2.], atol=1e-15)
    # float(2**53 + 1) equals float(2**53): raw-edge float signatures miss this.
    graph[0][1]["large"]["weight"] = 2**53
    np.testing.assert_array_equal(diffusion.relaxation_spectrum(graph), [0., 0.])
    graph[0][1]["large"]["weight"] -= 1
    with pytest.raises(ValueError, match="nonnegative"):
        diffusion.relaxation_spectrum(graph)
