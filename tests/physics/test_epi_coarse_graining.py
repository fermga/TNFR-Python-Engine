"""Exact and failed partition closure for the pure EPI nodal channel."""

import networkx as nx
import numpy as np
import pytest

from tnfr.physics import certify_epi_coarse_graining
from tnfr.physics.spectral_projectors import matrix_exponential


def _state(graph, epi, frequency):
    for node, value, nu_f in zip(graph, epi, frequency):
        graph.nodes[node].update(EPI=float(value), nu_f=float(nu_f), theta=0.0)
    return graph


def test_equitable_path_partition_closes_the_nodal_equation_numerically():
    graph = _state(nx.path_graph(4), [2.0, -1.0, 3.0, 0.5], [1.0] * 4)

    result = certify_epi_coarse_graining(graph, [(0, 3), (1, 2)])

    assert result.nodal_closure_within_tolerance
    assert result.morphism.nodal_flow_transport_within_tolerance
    assert result.projection_residual < 1e-12
    assert result.lift_residual < 1e-12
    assert result.information_loss_dimension == 2
    np.testing.assert_allclose(result.projection @ result.lift, np.eye(2))
    np.testing.assert_allclose(result.macro_frequency, [1.0, 0.5])


def test_exact_partition_transports_every_sampled_micro_trajectory():
    graph = _state(nx.path_graph(4), [2.0, -1.0, 3.0, 0.5], [1.0] * 4)
    result = certify_epi_coarse_graining(graph, [(0, 3), (1, 2)])
    state = np.array([2.0, -1.0, 3.0, 0.5])

    for time in [0.0, 0.1, 1.0, 4.0]:
        micro = matrix_exponential(-time * result.micro_generator) @ state
        macro = matrix_exponential(-time * result.macro_generator) @ (
            result.projection @ state
        )
        np.testing.assert_allclose(result.projection @ micro, macro, atol=1e-11)


def test_nonequitable_partition_records_unresolved_within_block_dynamics():
    graph = _state(nx.path_graph(5), np.arange(5.0), [1.0] * 5)

    result = certify_epi_coarse_graining(graph, [(0, 4), (1, 2, 3)])

    assert not result.nodal_closure_within_tolerance
    assert not result.morphism.nodal_flow_transport_within_tolerance
    assert result.projection_residual > 1e-3
    assert result.lift_residual > 1e-3
    assert result.information_loss_dimension == 3


def test_coarse_and_nested_morphism_use_one_relative_tolerance_semantics():
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(2, 3, weight=1.0)
    for source in (0, 1):
        for target in (2, 3):
            graph.add_edge(
                source,
                target,
                weight=1.000001 if (source, target) == (0, 2) else 1.0,
            )
    graph = _state(graph, np.arange(4.0), [1e6] * 4)

    result = certify_epi_coarse_graining(
        graph, [(0, 1), (2, 3)], tolerance=1e-3
    )

    # Absolute defects scale with the global nodal rate and exceed 1e-3, but
    # both identities have relative defects below the declared tolerance.
    assert result.projection_residual > 1e-3
    assert result.lift_residual > 1e-3
    assert result.relative_projection_residual < 1e-3
    assert result.relative_lift_residual < 1e-3
    assert result.nodal_closure_within_tolerance
    assert result.morphism.intertwines_within_tolerance
    assert result.morphism.relative_intertwining_residual == pytest.approx(
        result.relative_projection_residual
    )


def test_quotient_generator_retains_nodal_diffusion_form():
    graph = nx.cycle_graph(6)
    graph = _state(graph, np.arange(6.0), [0.5, 0.5, 1.5, 1.5, 0.5, 0.5])
    result = certify_epi_coarse_graining(graph, [(0, 1), (2, 3), (4, 5)])
    strength = result.macro_conductance.sum(axis=1)
    laplacian = np.diag(strength) - result.macro_conductance
    expected = (result.macro_frequency / strength)[:, None] * laplacian

    np.testing.assert_allclose(result.macro_generator, expected)
    np.testing.assert_allclose(result.macro_generator.sum(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "partition,message",
    [
        ([(0, 1, 2, 3)], "at least two"),
        ([(0, 1), (1, 2)], "exactly once"),
        ([(0,), (), (1, 2, 3)], "nonempty"),
    ],
)
def test_partition_validation_rejects_nonquotients(partition, message):
    graph = _state(nx.path_graph(4), np.arange(4.0), [1.0] * 4)
    with pytest.raises(ValueError, match=message):
        certify_epi_coarse_graining(graph, partition)


@pytest.mark.parametrize(
    "boolean", [True, False, np.bool_(True), np.bool_(False)]
)
def test_coarse_graining_rejects_boolean_tolerance(boolean):
    graph = _state(nx.path_graph(4), np.arange(4.0), [1.0] * 4)

    with pytest.raises(ValueError, match="tolerance.*not boolean"):
        certify_epi_coarse_graining(
            graph, [(0, 3), (1, 2)], tolerance=boolean
        )


@pytest.mark.parametrize("attribute", ["EPI", "nu_f"])
@pytest.mark.parametrize("boolean", [True, False, np.bool_(True), np.bool_(False)])
def test_coarse_graining_rejects_boolean_nodal_scalars(attribute, boolean):
    graph = _state(nx.path_graph(4), np.arange(4.0), [1.0] * 4)
    graph.nodes[0][attribute] = boolean

    with pytest.raises(ValueError, match="numeric, not boolean"):
        certify_epi_coarse_graining(graph, [(0, 3), (1, 2)])


def test_coarse_graining_rejects_zero_macro_capacity():
    graph = nx.Graph([(0, 1), (2, 3)])
    graph = _state(graph, np.arange(4.0), [1.0] * 4)

    with pytest.raises(ValueError, match="positive macro capacity"):
        certify_epi_coarse_graining(graph, [(0, 1), (2, 3)])


def test_coarse_graining_rejects_disconnected_macro_quotient():
    graph = nx.Graph([(0, 1), (2, 3), (4, 5), (6, 7)])
    graph = _state(graph, np.arange(8.0), [1.0] * 8)

    with pytest.raises(ValueError, match="connected macro quotient"):
        certify_epi_coarse_graining(
            graph,
            [(0, 2), (1, 3), (4, 6), (5, 7)],
        )
