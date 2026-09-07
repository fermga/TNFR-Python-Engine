"""Restricted metriplectic product of substrate pulse and EPI relaxation."""

import networkx as nx
import numpy as np
import pytest

from tnfr.physics import verify_metriplectic_product


def _state(graph, epi, frequency):
    for node, value, nu_f in zip(graph, epi, frequency):
        graph.nodes[node].update(
            EPI=float(value),
            nu_f=float(nu_f),
            theta=0.3 * node,
            delta_nfr=(-1.0) ** node * 0.2,
        )
    return graph


def test_product_preserves_hamiltonian_and_dissipates_dirichlet_energy():
    graph = _state(nx.path_graph(4), [2.0, -1.0, 3.0, 0.5], [0.4, 0.8, 1.2, 2.0])

    result = verify_metriplectic_product(graph)

    assert result.is_decoupled_metriplectic_bridge
    assert not result.couples_sectors
    assert result.state_dimension == 5 * len(graph)
    assert result.hamiltonian_derivative == pytest.approx(0.0, abs=1e-12)
    assert result.dirichlet_derivative < 0.0
    assert result.poisson_antisymmetry_residual < 1e-12
    assert result.minimum_dissipative_eigenvalue >= 0.0
    assert not result.stored_pressure_matches_epi_channel


def test_product_reports_when_stored_pressure_is_the_epi_channel():
    graph = _state(nx.path_graph(2), [0.0, 1.0], [0.5, 2.0])
    graph.nodes[0]["delta_nfr"] = 1.0
    graph.nodes[1]["delta_nfr"] = -1.0

    result = verify_metriplectic_product(graph)

    assert result.stored_pressure_matches_epi_channel
    assert result.stored_pressure_consistency_residual < 1e-12
    assert result.is_decoupled_metriplectic_bridge


def test_product_aligns_state_with_canonical_phase_space_node_order():
    graph = nx.Graph()
    graph.add_nodes_from([2, 1, 0])
    graph.add_edges_from([(2, 1), (1, 0)])
    for node in graph:
        graph.nodes[node].update(
            EPI=float(node),
            nu_f=1.0 + 0.25 * node,
            theta=0.3 * node,
            delta_nfr=(-1.0) ** node * 0.2,
        )

    result = verify_metriplectic_product(graph)

    assert result.nodes == (0, 1, 2)
    assert result.is_decoupled_metriplectic_bridge


def test_metriplectic_degeneracy_conditions_are_exact_block_identities():
    graph = _state(nx.cycle_graph(5), np.arange(5.0), np.linspace(0.5, 1.5, 5))

    result = verify_metriplectic_product(graph)

    assert result.poisson_dissipative_degeneracy == pytest.approx(0.0)
    assert result.metric_hamiltonian_degeneracy == pytest.approx(0.0)
    assert result.substrate_velocity_residual < 1e-12
    assert result.epi_velocity_residual < 1e-12


def test_uniform_epi_is_dissipative_equilibrium_while_pulse_can_continue():
    graph = _state(nx.path_graph(3), [7.0] * 3, [0.5, 1.0, 2.0])

    result = verify_metriplectic_product(graph)

    assert result.dirichlet_functional == pytest.approx(0.0)
    assert result.dirichlet_derivative == pytest.approx(0.0)
    assert np.linalg.norm(result.state_velocity[: 4 * len(graph)]) > 0.0
    np.testing.assert_allclose(result.state_velocity[4 * len(graph) :], 0.0)


def test_product_rejects_nonpositive_capacity_and_directed_transport():
    zero = _state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="positive finite capacity"):
        verify_metriplectic_product(zero)

    directed = _state(nx.DiGraph([(0, 1), (1, 0), (1, 2)]), [0.0, 1.0, 2.0], [1.0] * 3)
    with pytest.raises(ValueError, match="symmetric adjacency"):
        verify_metriplectic_product(directed)


@pytest.mark.parametrize("graph", [nx.Graph(), nx.empty_graph(1)])
def test_product_rejects_empty_or_singleton_support_explicitly(graph):
    with pytest.raises(ValueError, match="at least two supported nodes"):
        verify_metriplectic_product(graph)


@pytest.mark.parametrize("tolerance", [True, "1e-10", 1e-10j, None])
def test_product_rejects_nonreal_tolerance(tolerance):
    graph = _state(nx.path_graph(2), [0.0, 1.0], [1.0, 1.0])

    with pytest.raises(ValueError, match="finite and positive"):
        verify_metriplectic_product(graph, tolerance=tolerance)


@pytest.mark.parametrize("attribute", ["EPI", "nu_f", "theta", "delta_nfr"])
def test_product_rejects_boolean_nodal_channels(attribute):
    graph = _state(nx.path_graph(2), [0.0, 1.0], [1.0, 1.0])
    graph.nodes[0][attribute] = True

    with pytest.raises(ValueError, match="finite real"):
        verify_metriplectic_product(graph)


def test_product_rejects_nonrepresentable_finite_intermediates():
    graph = _state(nx.path_graph(2), [1e300, -1e300], [1.0, 1.0])
    graph.edges[0, 1]["weight"] = 1e100

    with pytest.raises(ValueError, match="floating-point range"):
        verify_metriplectic_product(graph)
